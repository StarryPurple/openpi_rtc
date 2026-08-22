from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

import pathlib

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    exp_name:str = "default",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        exp_name: 实验名称，用于配置动态加载
    """
    dc_list = []
    if len(config.combine_data) > 0:
        for i in range(len(config.combine_data)):
            dc_list.append(config.combine_data[i].create(
                (pathlib.Path(config.assets_base_dir)/config.combine_names[i]).resolve(), 
                config.model
                ))
            
        # 直接获取归一化后的数据并合并
        return create_combined_normalized_loader(
            dc_list,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
            exp_name=exp_name,
            sampling_weights=config.combine_batch,
        )
    
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        logging.info(f"data_config: {data_config}")

        if data_config.rlds_data_dir is not None:
            return create_rlds_data_loader(
                data_config,
                action_horizon=config.model.action_horizon,
                batch_size=config.batch_size,
                sharding=sharding,
                shuffle=shuffle,
                num_batches=num_batches,
                skip_norm_stats=skip_norm_stats,
                framework=framework,
            )
        return create_torch_data_loader(
            data_config,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )


def create_combined_normalized_loader(
    data_configs: list[_config.DataConfig],
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    skip_norm_stats: bool = False,
    framework: str = "jax",
    sampling_weights: list[float] | None = None,
    exp_name:str = "default",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建合并的归一化数据加载器
    
    直接处理多个data_config，获取归一化数据后合并为单一loader
    支持混合 Dataset（可索引）和 IterableDataset（如RLDS）
    
    优化策略：
    - 对于 RLDS 数据集，在创建时就分配好子批次大小，避免拆分重组的开销
    - 使用批次级别合并而非样本级别合并
    
    Args:
        data_configs: 数据配置列表
        model_config: 模型配置
        action_horizon: 动作序列长度
        batch_size: 总批次大小
        sharding: JAX sharding 配置
        shuffle: 是否打乱数据
        num_batches: 返回的批次数量限制
        seed: 随机种子
        skip_norm_stats: 是否跳过归一化统计
        framework: 使用的框架
        sampling_weights: 各数据源的采样权重（可选）。如果为 None，则均等分配。
        exp_name: 实验名称，用于配置动态加载
    """
    n_sources = len(data_configs)
    
    # 计算每个数据源的子批次大小
    if sampling_weights is not None and len(sampling_weights)>0:
        if len(sampling_weights) != n_sources:
            raise ValueError(
                f"sampling_weights length ({len(sampling_weights)}) must match "
                f"number of data sources ({n_sources})"
            )
        weights = np.array(sampling_weights, dtype=np.float64)
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_sources) / n_sources
    
    # 按权重分配子批次大小
    sub_batch_sizes = []
    remaining = batch_size
    for i in range(n_sources - 1):
        n = max(1, int(batch_size * weights[i]))
        n = min(n, remaining - (n_sources - i - 1))
        sub_batch_sizes.append(n)
        remaining -= n
    sub_batch_sizes.append(remaining)
    
    logging.info(f"Combined loader: batch_size={batch_size}, sub_batch_sizes={sub_batch_sizes}")
    
    # 创建各个数据集，保持原始顺序，记录类型
    # all_datasets: list of (dataset, sub_batch_size, is_iterable)
    all_datasets = []
    
    for idx, data_config in enumerate(data_configs):
        sub_batch = sub_batch_sizes[idx]
        
        if data_config.rlds_data_dir is not None:
            # RLDS 使用预计算的子批次大小
            dataset = create_rlds_dataset(
                data_config, action_horizon, sub_batch, shuffle=shuffle
            )
            normalized = transform_iterable_dataset(
                dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True
            )
            all_datasets.append((normalized, sub_batch, True))  # is_iterable=True
        else:
            dataset = create_torch_dataset(data_config, action_horizon, model_config)
            normalized = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
            all_datasets.append((normalized, sub_batch, False))  # is_iterable=False
    
    # 创建组合数据加载器
    return MergedNormalizedDataLoader(
        all_datasets=all_datasets,
        data_configs=data_configs,
        batch_size=batch_size,
        sharding=sharding,
        num_batches=num_batches,
        num_workers=num_workers,
        framework=framework,
        seed=seed,
        shuffle=shuffle,
        exp_name=exp_name,
    )


class MergedNormalizedDataLoader(DataLoader):
    """合并多个归一化数据集的数据加载器
    
    支持混合可索引数据集 (Dataset) 和可迭代数据集 (IterableDataset/RLDS)
    
    优化策略：
    - 批次级别合并：每个数据源直接产出子批次，然后合并
    - 保持数据源的原始顺序，确保采样权重与配置一致
    - RLDS 数据集创建时就使用预计算的子批次大小
    """
    
    def __init__(
        self,
        all_datasets: list[tuple[Dataset | IterableDataset, int, bool]] | None = None,
        data_configs: list[_config.DataConfig] | None = None,
        batch_size: int = 32,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        framework: str = "jax",
        seed: int = 0,
        shuffle: bool = True,
        exp_name:str = "default",
    ):
        """
        Args:
            all_datasets: 数据集列表，每项为 (dataset, sub_batch_size, is_iterable)
                保持与原始 data_configs 相同的顺序
            batch_size: 总批次大小
            sharding: JAX sharding 配置
            num_batches: 返回的批次数量限制
            num_workers: 数据加载的 worker 进程数
            framework: 使用的框架 ("jax" 或 "pytorch")
            seed: 随机种子
            shuffle: 是否打乱批次内数据
            exp_name: 实验名称，用于配置动态加载
        """
        self._all_datasets = all_datasets or []
        self._data_configs = data_configs or []
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._framework = framework
        self._shuffle = shuffle
        self._seed = seed
        self.exp_name=exp_name
        
        # 设置 sharding
        if sharding is None and framework == "jax":
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding
        
        # 数据源总数
        self._n_sources = len(self._all_datasets)
        
        if self._n_sources == 0:
            raise ValueError("At least one dataset must be provided.")
        
        # 为每个可索引数据源创建 torch DataLoader（多进程并行加载）
        n_indexable = sum(1 for _, _, is_it in self._all_datasets if not is_it)
        workers_per_source = max(1, num_workers // max(n_indexable, 1)) if num_workers > 0 and n_indexable > 0 else 0
        
        self._source_loaders:list[torch.utils.data.DataLoader | IterableDataset] = []
        for dataset, sub_batch_size, is_iterable in self._all_datasets:
            if is_iterable:
                self._source_loaders.append(dataset)  # 直接用作可迭代对象
            else:
                mp_context = None
                if workers_per_source > 0:
                    mp_context = multiprocessing.get_context("spawn")
                generator = torch.Generator()
                generator.manual_seed(seed)
                torch_loader = torch.utils.data.DataLoader(
                    typing.cast(torch.utils.data.Dataset, dataset),
                    batch_size=sub_batch_size,
                    shuffle=shuffle,
                    num_workers=workers_per_source,
                    multiprocessing_context=mp_context,
                    persistent_workers=workers_per_source > 0,
                    collate_fn=_collate_fn,
                    worker_init_fn=_worker_init_fn,
                    drop_last=True,
                    generator=generator,
                )
                self._source_loaders.append(torch_loader)
        
        # 日志
        sub_batches_info = [(s, "iter" if is_iter else f"idx/w{workers_per_source}") for _, s, is_iter in self._all_datasets]
        logging.info(f"MergedNormalizedDataLoader: sources={sub_batches_info}")

    def data_config(self) -> _config.DataConfig:
        """返回第一个数据源的配置，供 CheckpointManager 获取 norm_stats 使用"""
        if self._data_configs:
            # return self._all_datasets[0][0].data_config() if hasattr(self._all_datasets[0][0], 'data_config') else self._data_configs[0]
            if hasattr(self._all_datasets[0][0], 'data_config'):
                return self._all_datasets[0][0].data_config()  # type: ignore
            else:
                return self._data_configs[0]
        return None

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        
        # 为每个数据源创建无限迭代器（epoch 耗尽后自动重建，与 TorchDataLoader 行为一致）
        source_iters = [iter(loader) for loader in self._source_loaders]
        
        produced = 0
        while True:
            if self._num_batches is not None and produced >= self._num_batches:
                return
            
            sub_batches = []
            for i in range(self._n_sources):
                try:
                    sub_batch = next(source_iters[i])
                except StopIteration:
                    source_iters[i] = iter(self._source_loaders[i])
                    sub_batch = next(source_iters[i])
                ########## HACK: mask certain images ################
                if "mask" in self.exp_name:
                    batch_config = self._data_configs[i]
                    if 'human' in batch_config.repo_id:
                        sub_batch['image_mask']['left_wrist_0_rgb']=np.full(sub_batch['image_mask']['left_wrist_0_rgb'].shape, False, dtype=bool)
                        sub_batch['image_mask']['right_wrist_0_rgb']=np.full(sub_batch['image_mask']['right_wrist_0_rgb'].shape, False, dtype=bool)
                #####################################################
                sub_batches.append(sub_batch)
            
            # 合并子批次
            if len(sub_batches) == 1:
                batch = sub_batches[0]
            else:
                if self._shuffle:
                    perm = rng.permutation(self._batch_size)
                    batch = jax.tree.map(
                        lambda *xs: np.concatenate(xs, axis=0)[perm],
                        *sub_batches,
                    )
                else:
                    batch = jax.tree.map(
                        lambda *xs: np.concatenate(xs, axis=0),
                        *sub_batches,
                    )
            
            produced += 1
            
            # 应用 sharding
            if self._sharding is not None:
                batch = jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch
                )
            elif self._framework == "pytorch":
                batch = jax.tree.map(torch.as_tensor, batch)
            
            yield (
                _model.Observation.from_dict(batch),
                batch["actions"],
            )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
