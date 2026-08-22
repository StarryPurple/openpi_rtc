import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def make_r1pro_example() -> dict:
    """为 R1Pro 策略创建随机输入示例（19维）。"""
    return {
        # 19维状态: 3(底盘) + 7(左臂) + 1(左爪) + 7(右臂) + 1(右爪)
        "observation/state": np.random.rand(19),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Move the box",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3: # 处理 LeRobot 的 [C, H, W] 格式
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class R1ProInputs(transforms.DataTransformFn):
    """
    R1Pro 输入转换类。将数据集中的 19 维数据和多摄像头画面转换为 Pi0 模型格式。
    """
    # 模型动作维度（用于 padding）
    action_dim: int = 19
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # 1. 状态处理: 将 19 维 proprioceptive 输入 pad 到模型动作维度
        # 如果数据集中的 key 不同，请修改 "observation/state"
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # 2. 图像处理: 适配 R1Pro 的三个视角
        # 主视角 (第三人称)
        base_image = _parse_image(data["observation/image"])
        # 左手腕视角
        left_wrist_image = _parse_image(data.get("observation/left_wrist_image", np.zeros_like(base_image)))
        # 右手腕视角
        right_wrist_image = _parse_image(data.get("observation/right_wrist_image", np.zeros_like(base_image)))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if "observation/left_wrist_image" in data else np.False_,
                "right_wrist_0_rgb": np.True_ if "observation/right_wrist_image" in data else np.False_,
            },
        }

        # 如果是 pi0 模型且没有图像，应用 padding mask
        if mask_padding:
            for k in inputs["image_mask"]:
                if not inputs["image_mask"][k]:
                    inputs["image"][k] = np.zeros_like(base_image)

        # 3. 动作处理 (仅训练阶段)
        if "actions" in data:
            # 将 19 维动作 pad 到模型动作维度
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # 4. 指令处理
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class R1ProOutputs(transforms.DataTransformFn):
    """
    R1Pro 输出转换类。将模型输出转换回 R1Pro 特定的 19 维控制格式。
    """
    def __call__(self, data: dict) -> dict:
        # 核心逻辑：只截取前 19 个维度
        # 对应：底盘(0-2), 左臂(3-9), 右臂(10-16) , 左爪(17), 右爪(18)
        return {"actions": np.asarray(data["actions"][:, :19])}