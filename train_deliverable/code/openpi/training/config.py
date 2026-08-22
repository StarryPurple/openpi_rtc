"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.R1pro_policy as r1pro_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(
                                model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(
                                model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(
                self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(
                _download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(
                f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(
                model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(
                model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(
                model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
@dataclasses.dataclass(frozen=True)
class LeRobotR1ProDataConfig(DataConfigFactory):
    """
    R1Pro 搬箱子机器人专用的数据转换配置类。
    适配 19 维动作空间：底盘(3) + 左臂(7) + 左爪(1) + 右臂(7) + 右爪(1)
    """
    default_prompt: str | None = None
    
    action_sequence_keys: Sequence[str] = ("action",)
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. Repack Transform: 仅用于训练集数据预处理。
        # 将数据集中的 Key 映射为 R1Pro 策略类需要的 Key 名字。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action", 
                    }
                )
            ]
        )

        # 2. Data Transforms: 核心转换逻辑。
        # 使用我们之前定义的 R1Pro 专用 Inputs 和 Outputs 类。
        data_transforms = _transforms.Group(
            inputs=[r1pro_policy.R1ProInputs(
                action_dim=model_config.action_dim, 
                model_type=model_config.model_type
            )],
            outputs=[r1pro_policy.R1ProOutputs()],
        )

        # 3. Delta Action Mask (增量动作掩码):
        delta_action_mask = _transforms.make_bool_mask(17, 2)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
        
        # 4. Model Transforms: 提示词 Token 化等，保持默认即可。
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        # 返回完整配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(
        default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    
    combine_data: list[DataConfigFactory] = dataclasses.field(default_factory=list)
    combine_names: list[str] = dataclasses.field(default_factory=list)
    combine_batch: list[int] = dataclasses.field(default_factory=list)


    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints/qb-ilm-ckpts/g100_pi"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ============================================================================
# XTrainer Task Config Generator
# ============================================================================
MODEL_DICTS = {
    # base model
    "pi0": "gs://openpi-assets/checkpoints/pi0_base/params",
    "pi05": "gs://openpi-assets/checkpoints/pi05_base/params",
    # agibot pretrain
    "pi_agibot800_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot_pretrain_from_paligemma/agibot800_paligemma/340000/params",
    "pi_agibot3k_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_multinode/checkpoints/pi0_agibot_pretrain_from_paligemma_3kh_restore/pi0_agibot_pretrain_from_paligemma_3kh_restore/60000/params",
    "pi_agibot8_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot_pretrain_from_paligemma_8h/agibot_8h/120000/params",
    "pi_agibot80_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot_pretrain_from_paligemma_80h_restore/agibot_80h_restore_130k/60000/params",
    
    # human pretrain
    "pi_human8_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_human_pretrain_from_paligemma_8h/pi0_human_pretrain_from_paligemma_8h/120000/params",
    "pi_human800_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi/checkpoints/pi0_human_pretrain_from_paligemma/pi0_human_pretrain_from_paligemma_new/340000/params",
    "pi_human800_paligemma_200k": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi/checkpoints/pi0_human_pretrain_from_paligemma/pi0_human_pretrain_from_paligemma_new/200000/params",
    "pi_human80_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_human_pretrain_from_paligemma_80h/pi0_human_pretrain_from_paligemma_80h/180000/params",
    # "pi_human_ego4d_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_multinode/checkpoints/pi0_ego4d_pretrain_from_paligemma/pi0_ego4d_pretrain_from_paligemma-bz2048/70000/params",
    # cotrain"s3://openpi-assets/checkpoints/pi0_base/params"
    "pi_agibot800_egodex_cotrain": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot_pretrain_from_human_340k/pi0_agibot_pretrain_from_human_340k/300000/params",
    "pi_agibot800_egodex200k_cotrain": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot_pretrain_from_human/pi0_agibot_pretrain_from_human_1127/300000/params",
    "pi_agibot3k_egodex_cotrain": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_multinode/checkpoints/pi0_agibot3k_cotrain_from_human/pi0_agibot3k_cotrain_from_human/90000/params",
    "pi_agibot3k_egodex200k_cotrain": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_multinode/checkpoints/pi0_agibot3k_cotrain_from_human_200k/pi0_agibot3k_cotrain_from_human_200k_1127/90000/params",
    "pi_agibot80_egodex_cotrain": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_agibot80_pretrain_from_human_340k/pi0_agibot80_pretrain_from_human_340k/180000/params",
    # nostate
    "pi_nostate_human800_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_no_state_human_pretrain_from_paligemma_800h/nostate_human_800/160000/params",
    "pi_nostate_human80_paligemma": "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/ziyu/openpi_pretrain/checkpoints/pi0_no_state_human_pretrain_from_paligemma_80h/nostate_human_80/200000/params",

}


@dataclasses.dataclass(frozen=True)
class XTrainerTaskConfig:
    """Configuration for an XTrainer task."""
    # Task name (e.g., "pouring_gravel", "cut_playdough")
    task_name: str
    # Dataset repo_id for training (if different from task_name)
    train_repo_id: str | None = None
    # Dataset repo_id for evaluation (if different from task_name + "_eval")
    eval_repo_id: str | None = None
    # Asset directory base path (relative to the repository root; the
    # standalone bundle keeps everything machine-independent).
    assets_base_dir: str = "./assets"
    # Training hyperparameters
    batch_size: int = 32
    num_train_steps: int = 50000
    keep_period: int = 30_000 #"s3://openpi-assets/checkpoints/pi0_base/params"
    fsdp_devices: int = 1
    num_workers: int = 12
    combine_list: list[str] = dataclasses.field(default_factory=list)
    combine_batch: list[int] = dataclasses.field(default_factory=list)

    @property
    def train_dataset(self) -> str:
        """Get the training dataset repo_id."""
        return self.train_repo_id or self.task_name

    @property
    def eval_dataset(self) -> str:
        """Get the evaluation dataset repo_id."""
        return self.eval_repo_id or f"{self.task_name}_eval"


def _create_xtrainer_repack_transforms() -> _transforms.Group:
    """Create standard repack transforms for XTrainer tasks."""
    return _transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })
    ])


def create_xtrainer_config(
    task: XTrainerTaskConfig,
    model_type: str,
    mode: Literal["train", "eval"] = "train",
) -> TrainConfig:
    """
    Create a TrainConfig for an XTrainer task.

    Args:
        task: The task configuration.
        model_type: Model type, either "pi0" or "pi05".
        mode: Either "train" or "eval".

    Returns:
        A TrainConfig instance.

    Example:
        >>> task = XTrainerTaskConfig(task_name="pouring_gravel", train_repo_id="pouring_gravel_test")
        >>> config = create_xtrainer_config(task, model_type="pi05", mode="train")
    """
    # Config name format: {model_type}-{task_name}-xtrainer[-eval]
    train_config_name = f"{model_type}-{task.task_name}-xtrainer"
    eval_config_name = f"{train_config_name}-eval"
    if mode == "eval":
        config_name = eval_config_name
    else:
        config_name = train_config_name

    # Determine dataset repo_id based on mode
    repo_id = task.train_dataset if mode == "train" else task.eval_dataset
    # 这里需要注意，为了保证 eval 时的公平性，不能直接把 eval set 的 norm stats 给到 dataloader， 要给训练集的
    assets_repo_id = task.train_dataset

    # Determine model config and weight loader based on model_type
    if model_type == "pi05":
        model = pi0_config.Pi0Config(pi05=True)
        if task.task_name == "task_00026op":
            weight_loader_instance = weight_loaders.CheckpointWeightLoader(
                "/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/dijiang/49999/params"
            )
        else:
            weight_loader_instance = weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ) 
    else:  # pi0-type
        model = pi0_config.Pi0Config()
        weight_loader_instance = weight_loaders.CheckpointWeightLoader(
            MODEL_DICTS[model_type]
        )

    # Asset directory
    assets_dir = f"{task.assets_base_dir}/pi0-{task.task_name}-xtrainer"

    return TrainConfig(
        name=config_name,
        model=model,
        data=LeRobotAlohaDataConfig(
            repo_id=repo_id,
            adapt_to_pi=False,
            repack_transforms=_create_xtrainer_repack_transforms(),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir=assets_dir,
                asset_id=assets_repo_id,
            ),
        ),
        batch_size=task.batch_size,
        weight_loader=weight_loader_instance,
        num_train_steps=task.num_train_steps,
        keep_period=task.keep_period,
        save_interval=10000,
        fsdp_devices=task.fsdp_devices,
        num_workers=task.num_workers,
    )
    
def create_combine_xtrainer_config(
    task: XTrainerTaskConfig,
    model_type: str,
    mode: Literal["train", "eval"] = "train",
) -> TrainConfig:
    
    if len(task.combine_list) == 0:
        return create_xtrainer_config(task, model_type, mode)
    
    train_config_name = f"{model_type}-{task.task_name}-xtrainer"
    eval_config_name = f"{train_config_name}-eval"
    if mode == "eval":
        config_name = eval_config_name
    else:
        config_name = train_config_name
    
    # load subtasks' dataConfig
    datalist = []
    
    for STask_name in task.combine_list:
        STaskConfig = None
        for tc in _CONFIGS:
            if STask_name in tc.name:
                STaskConfig = tc
                break
        if STaskConfig is not None:
            datalist.append(STaskConfig.data)
            
    # Determine model config and weight loader based on model_type
    if model_type == "pi05":
        model = pi0_config.Pi0Config(pi05=True)
        weight_loader_instance = weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        )
    else:  # pi0-type
        model = pi0_config.Pi0Config()
        weight_loader_instance = weight_loaders.CheckpointWeightLoader(
            MODEL_DICTS[model_type]
        )
        
    return TrainConfig(
        name=config_name,
        model=model,
        data=datalist[0],
        combine_data=datalist,
        combine_names=task.combine_list,
        combine_batch=task.combine_batch,
        batch_size=task.batch_size,
        weight_loader=weight_loader_instance,
        num_train_steps=task.num_train_steps,
        keep_period=task.keep_period,
        save_interval=10000,
        fsdp_devices=task.fsdp_devices,
        num_workers=task.num_workers,
    )
        


def generate_xtrainer_configs(tasks: list[XTrainerTaskConfig]) -> list[TrainConfig]:
    """
    Generate all config combinations for a list of XTrainer tasks.

    For each task, generates:
    - {pi0|pi05}-{task_name}-xtrainer (training)
    - {pi0|pi05}-{task_name}-xtrainer-eval (evaluation)

    Args:
        tasks: List of task configurations.

    Returns:
        List of TrainConfig instances.
    """
    configs = []
    for task in tasks:
        for model_type in MODEL_DICTS:
            for mode in ["train", "eval"]:
                configs.append(create_xtrainer_config(task, model_type, mode))
    return configs

def generate_combine_xtrainer_configs(tasks: list[XTrainerTaskConfig]) -> list[TrainConfig]:
    configs = []
    for task in tasks:
        for model_type in MODEL_DICTS:
            for mode in ["train", "eval"]:
                configs.append(create_combine_xtrainer_config(task, model_type, mode))
    return configs



# ============================================================================
# Define your XTrainer tasks here
# ============================================================================
XTRAINER_TASKS = [
    XTrainerTaskConfig(
        task_name="task_00003",
        train_repo_id="task_00003_train",
        eval_repo_id="task_00003_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00005",
        train_repo_id="task_00005_train",
        eval_repo_id="task_00005_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00006",
        train_repo_id="task_00006_train",
        eval_repo_id="task_00006_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00008",
        train_repo_id="task_00008_train",
        eval_repo_id="task_00008_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00009",
        train_repo_id="task_00009_train",
        eval_repo_id="task_00009_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00014",
        train_repo_id="task_00014_train",
        eval_repo_id="task_00014_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00018",
        train_repo_id="task_00018_train",
        eval_repo_id="task_00018_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00026",
        train_repo_id="task_00026_train",
        eval_repo_id="task_00026_eval",
    ),
    # BEGIN AUTO MIX CONFIG task_00026_mix50
    XTrainerTaskConfig(
        task_name="task_00026_mix50",
        train_repo_id="task_00026_mix50_train",
        eval_repo_id="task_00026_mix50_eval",
    ),
    # END AUTO MIX CONFIG task_00026_mix50
    XTrainerTaskConfig(
        task_name="task_00031",
        train_repo_id="task_00031_train",
        eval_repo_id="task_00031_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00031_mingwei",
        train_repo_id="tube_xtrainer_mingwei",
        eval_repo_id="task_00031_mingwei_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00030",
        train_repo_id="task_00030_train",
        eval_repo_id="task_00030_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00038",
        train_repo_id="task_00038_train",
        eval_repo_id="task_00038_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00121",
        train_repo_id="task_00121_train",
        eval_repo_id="task_00121_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00020",
        train_repo_id="task_00020_train",
        eval_repo_id="task_00020_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00011",
        train_repo_id="task_00011_train",
        eval_repo_id="task_00011_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_yushun_fry",
        train_repo_id="task_yushun_fry_train",
    ),
    XTrainerTaskConfig(
        task_name="task_yushun_pot_lid",
        train_repo_id="task_yushun_pot_lid_train",
    ),
    XTrainerTaskConfig(
        task_name="task_push_drawer_human",
        train_repo_id="task_push_drawer_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00005_human",
        train_repo_id="task_00005_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00006_human",
        train_repo_id="task_00006_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00011_human",
        train_repo_id="task_00011_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00018_human",
        train_repo_id="task_00018_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00026_human",
        train_repo_id="task_00026_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00031_human",
        train_repo_id="task_00031_human_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00026_RT-easy",
        train_repo_id="single-blocks_ranking_size-easy",
    ),
    XTrainerTaskConfig(
        task_name="task_00026_RT-hard",
        train_repo_id="single-blocks_ranking_size-hard",
    ),
    XTrainerTaskConfig(
        task_name="task_00003_zehao",
        train_repo_id="task_00003_zehao_train",
        eval_repo_id="task_00003_zehao_eval",
    ),
    XTrainerTaskConfig(
        task_name="pouring_gravel_donglin",
        train_repo_id="pouring_gravel_donglin_train",
        eval_repo_id="pouring_gravel_donglin_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00031_jy",
        train_repo_id="jy_tube_train",
        eval_repo_id="jy_tube_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00001_jy_erase",
        train_repo_id="jy_erase_train",
        eval_repo_id="jy_erase_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_sort_cubes_base",
        train_repo_id="sort_cubes_base_train",
    ),
    XTrainerTaskConfig(
        task_name="task_00026op",
        train_repo_id = "task_00026op_train",
    ),
    XTrainerTaskConfig(
    task_name="task_00031_light",
    train_repo_id="task_00031_light_train",
    eval_repo_id="task_00031_light_eval",
    ),
    XTrainerTaskConfig(
    task_name="task_00031_new",
    train_repo_id="task_00031_new_train",
    eval_repo_id="task_00031_new_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00031_yulong_copy",
        train_repo_id="task_00031_yulong_copy_train",
        eval_repo_id="task_00031_yulong_copy_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_cubeinbowl",
        train_repo_id="task_zehao_cubeinbowl_train",
        #eval_repo_id="task_zehao_cubeinbowl_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowl/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowl/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_cubeinbowlwoslow",
        train_repo_id="task_zehao_cubeinbowlwoslow_train",
        #eval_repo_id="task_zehao_cubeinbowl_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowl/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowl/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_cubeinbowlfast",
        train_repo_id="task_zehao_cubeinbowlfast_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
     XTrainerTaskConfig(
        task_name="task_zehao_cubeinbowlslow",
        train_repo_id="task_zehao_cubeinbowlslow_train",
        eval_repo_id="task_zehao_cubeinbowlslow_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlslow/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlslow/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_teartapesmall",
        train_repo_id="task_zehao_teartapesmall_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_teartapebig",
        train_repo_id="task_zehao_teartapebig_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stack",
        train_repo_id="task_zehao_stack_train",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stackbowl",
        train_repo_id="task_zehao_stackbowl_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stackbigcup",
        train_repo_id="task_zehao_stackbigcup_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stackpapercup",
        train_repo_id="task_zehao_stackpapercup_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stackflightcup",
        train_repo_id="task_zehao_stackflightcup_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stackplasticcup",
        train_repo_id="task_zehao_stackplasticcup_train",
        #eval_repo_id="task_zehao_cubeinbowlfast_eval",
        #task_description="Put the cube into the bowl.",
        #train_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/train",
        #eval_raw_dir="/inspire/qb-ilm/project/robot-reasoning/public/RHOS/zehao/dataset_zehao_cubeinbowlfast/eval",
    ),
    #Add more tasks here...
    # XTrainerTaskConfig(task_name="new_task", train_repo_id="new_task_train"),
]

COMBINE_XTRAINER_TASKS = [
    XTrainerTaskConfig(
        task_name="task_yushun_combine",
        combine_list=[
            "task_yushun_fry",
            "task_yushun_pot_lid",
        ],
        fsdp_devices=2,
        batch_size=32
    ),
    XTrainerTaskConfig(
        task_name="task_push_drawer_combine",
        combine_list=[
            "task_00003",
            "task_push_drawer_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00005_combine",
        combine_list=[
            "task_00005",
            "task_00005_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00006_combine",
        combine_list=[
            "task_00006",
            "task_00006_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00011_combine",
        combine_list=[
            "task_00011",
            "task_00011_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00018_combine",
        combine_list=[
            "task_00018",
            "task_00018_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00026_combine",
        combine_list=[
            "task_00026",
            "task_00026_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00031_combine",
        combine_list=[
            "task_00031",
            "task_00031_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00026_RT-easy_combine",
        combine_list=[
            "task_00026_RT-easy",
            "task_00026_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00026_RT-hard_combine",
        combine_list=[
            "task_00026_RT-hard",
            "task_00026_human",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00121_task_00009_combine",
        combine_list=[
            "task_00121",
            "task_00009",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00121_task_00018_combine",
        combine_list=[
            "task_00121",
            "task_00018",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00038_task_00018_combine",
        combine_list=[
            "task_00038",
            "task_00018",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00038_task_00006_combine",
        combine_list=[
            "task_00038",
            "task_00006",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00001_task_00026_combine",
        combine_list=[
            "task_00001_jy_erase",
            "task_00026",
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00018_all_combine",
        combine_list=[
            "task_00018",
            "task_push_drawer_human",
            "task_00006_human",
            "task_00011_human",
            "task_00018_human",
            "task_00026_human",
            "task_00031_human",
        ],
        combine_batch=[20,2,2,2,2,2,2],
        fsdp_devices=2,
        batch_size=32,
        num_workers=42, # dataset_num * 6 to maximize 2 H200
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_teartampesmall_combine",
        combine_list=[
            "task_zehao_teartapesmall"
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_combine",
        combine_list=[
            "task_zehao_cubeinbowl",
            "task_zehao_cubeinbowlslow",
            "task_zehao_cubeinbowlfast"
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_combine_wo_slow",
        combine_list=[
            "task_zehao_cubeinbowl",
            "task_zehao_cubeinbowlfast"
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_combine_wo_fast",
        combine_list=[
            "task_zehao_cubeinbowl",
            "task_zehao_cubeinbowlslow"
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
    XTrainerTaskConfig(
        task_name="task_00031_yulong",
        train_repo_id="task_00031_yulong_train",
        eval_repo_id="task_00031_yulong_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_00031_entong",
        train_repo_id="task_00031_entong_train",
        eval_repo_id="task_00031_entong_eval",
    ),
    XTrainerTaskConfig(
        task_name="task_zehao_stack_combine",
        combine_list=[
            "task_zehao_stackbowl",
            "task_zehao_stackbigcup",
            "task_zehao_stackpapercup",
            "task_zehao_stackplasticcup",
            "task_zehao_stackbigcup"
        ],
        fsdp_devices=2,
        batch_size=32,
    ),
]


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_dobot_cy_small_tube",
        model=pi0_config.Pi0Config(
            max_token_len=300,
            pi05=True),
        batch_size=64,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi = False,
            # repo_id="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/tool_adaptation_100",
            repo_id="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/small_tube_one.0",
            
            # assets=AssetsConfig(
            #     assets_dir="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/task0063_dataset",
            #     asset_id="tube",
            # ),
            default_prompt="Move the small tube into another rack",

            # default_prompt="align the 2 line segments with the same color",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.top",
                                "cam_left_wrist": "observation.images.left_wrist",
                                "cam_right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_dobot_zy_small_tube",
        model=pi0_config.Pi0Config(
            max_token_len=300,
            pi05=True),
        batch_size=32,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi = False,
            # repo_id="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/tool_adaptation_100",
            repo_id="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/small_tube_one",
            
            # assets=AssetsConfig(
            #     assets_dir="/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/cy/datasets/task0063_dataset",
            #     asset_id="tube",
            # ),
            default_prompt="Move the small tube into another rack",

            # default_prompt="align the 2 line segments with the same color",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.top",
                                "cam_left_wrist": "observation.images.left_wrist",
                                "cam_right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
     TrainConfig(
        name="pi05-moving_box-xtrainer",
        model=pi0_config.Pi0Config(max_token_len=300,
            pi05=True),
        data=LeRobotR1ProDataConfig(          # 2. 核心修改：换成了 R1Pro 专用配置类
            repo_id="movebox",
            default_prompt="Move the box",
            base_config=DataConfig(
                prompt_from_task=False,        
            ),
        ),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        save_interval=10_000,
        keep_period=10_000,
        fsdp_devices=1,  
        num_workers=12,
    ),
     
    TrainConfig(
        name="pi05_bussingtable_high",
        model=pi0_config.Pi0Config(
            max_token_len=300,
            pi05=True),
        batch_size=64,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi = False,
            
            repo_id="bussingtable_high",
            
            assets=AssetsConfig(
                assets_dir="./assets/pi05_bussingtable_high",
                asset_id="bussingtable_high",
            ),
            default_prompt="Pick and place the syringe, tape measure, and spatula into the checkered box, then the paper cup and Velcro roll into the transparent box.",
            
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
     TrainConfig(
        name="pi05_bussingtable_mid",
        model=pi0_config.Pi0Config(
            max_token_len=300,
            pi05=True),
        batch_size=64,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi = False,
            
            repo_id="bussingtable_mid",
            
            assets=AssetsConfig(
                assets_dir="./assets/pi05_bussingtable_mid",
                asset_id="bussingtable_mid",
            ),
            default_prompt="Pick and place the syringe, tape measure, and spatula into the checkered box, then the paper cup and Velcro roll into the transparent box.",
            
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
     TrainConfig(
        name="pi05_bussingtable_low",
        model=pi0_config.Pi0Config(
            max_token_len=300,
            pi05=True),
        batch_size=64,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi = False,
            
            repo_id="bussingtable_low",
            
            assets=AssetsConfig(
                assets_dir="./assets/pi05_bussingtable_low",
                asset_id="bussingtable_low",
            ),
            default_prompt="Pick and place the syringe, tape measure, and spatula into the checkered box, then the paper cup and Velcro roll into the transparent box.",
            
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
     
    # Debug config
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05-base",
        model=pi0_config.Pi0Config(
            pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="trossen",
            adapt_to_pi=False,
            repack_transforms=None,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                asset_id="trossen",
            ),
        ),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="pi05-base",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi0-paligemma",
        model=pi0_config.Pi0Config(
            pi05=False),
        data=LeRobotAlohaDataConfig(
            repo_id="task_00003_train",
            adapt_to_pi=False,
            repack_transforms=_create_xtrainer_repack_transforms(),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="./assets/pi0-task_00003-xtrainer",
                asset_id="task_00003_train",
            ),
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="pi0-paligemma",
        wandb_enabled=False,
    ),
    
    # ── A2-CAP: per-episode prompt, π0.5 backbone ──
    # 使用独立 config name 避免与 XTRAINER_TASKS 自动生成的同名 config 冲突
    TrainConfig(
        name="pi05-a2_cap_v1-xtrainer",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="task_zixuan_a2_cap_v1_train",
            adapt_to_pi=False,
            repack_transforms=_create_xtrainer_repack_transforms(),
            base_config=DataConfig(
                prompt_from_task=True,  # A2-CAP 使用 per-episode prompt
            ),
        ),
        batch_size=32,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=50000,
        keep_period=30_000,
        fsdp_devices=1,
        num_workers=12,
    ),
]

# Auto-generate all XTrainer configs
_XTRAINER_CONFIGS = generate_xtrainer_configs(XTRAINER_TASKS)
_CONFIGS += _XTRAINER_CONFIGS  # Add auto-generated XTrainer configs
_COMBINE_XTRAINER_CONFIGS = generate_combine_xtrainer_configs(COMBINE_XTRAINER_TASKS)
_CONFIGS += _COMBINE_XTRAINER_CONFIGS


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
