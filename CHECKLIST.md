# 环境检查与验收清单（工控机 / 推理机 / GPU 机器）

机器分工（已确认）：

- **推理机（GPU 机器）**：训练 + 离线评估 + probe。与当前开发机共享文件系统。
- **工控机**：实机执行——**自己加载已经训练好的 checkpoint，实时跑模型推理
  并驱动机器人**（RTX 4090 / 24GB，Python 3.13.5）。需要装好 openpi + jax
  (CUDA) + cv2 + pyrealsense2 + dobot_control；不需要任何训练功能。
- 三方法（inference-RTC / train-RTC / πR²）的模型都是训练好的 checkpoint，
  实机对比时在工控机上逐个加载。

数据约定：

- 参考 checkpoint（baseline / 微调起点）：yulong 数据训练出的 49999。
- 微调数据：`task_00031_entong`（约定与 yulong 完全一致：left-first 14 维、
  三路相机、delta 动作、夹爪 1=开 0=闭）。
- 两个 task config 都已注册：`pi05-task_00031_yulong-xtrainer`（49999 /
  baseline）与 `pi05-task_00031_entong-xtrainer`（entong 微调与评测，默认）。

```bash
export OPENPI05_CHECKPOINT_49999=/inspire/qb-ilm/project/robot-reasoning/xuyue-p-xuyue/ziyu/checkpoints/g100_pi/pi05-task_00031_yulong-xtrainer/default_pi05/49999
export OPENPI05_RAW_TRAIN_DIR=/inspire/hdd/project/robot-reasoning/public/RHOS/dobot/task_00031_entong/train
```

---

## 阶段 0：GPU 机器（训练/评估）

```bash
bash run_probe.sh        # 需要 OPENPI05_RAW_TRAIN_DIR（或第二个参数）；输出 probe_report_*.txt
```

通过后：

```bash
uv run python eval_offline_rtc.py --mode baseline \
  --config pi05-task_00031_yulong-xtrainer \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR"
```

## 阶段 1：工控机环境检查（一条命令）

工控机 Python 是 3.8.10，且没有 uv。环境分两层：

- **ACT / 控制栈**（`launch_nodes.py` / `run_inference.py`）：用系统
  Python 3.8.10，平台自带依赖，`python3.8` 直接跑；
- **openpi 栈**（`run_robot.py` / `measure_latency.py`）：要求
  Python ≥3.11，用 uv 管理（openpi 的 jax/flax/transformers 不支持 3.8，
  测试条件不能降到 3.8）。先在工控机装 uv（单二进制）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # 或拷 uv 二进制到 PATH
export PATH="$HOME/.local/bin:$PATH"
cd <rtc_control> && uv sync                          # 自动下载/使用 Python 3.11 并装依赖
```

内网不能直连 pypi 时，给 uv 配镜像（如
`UV_PYTHON_INSTALL_MIRROR` / `PIP_INDEX_URL` 指向内网源）后重跑 `uv sync`。

```bash
# 在工控机上运行（dobot_control 仓库根目录，或加 --control-repo 指定）
uv run python check_robot_pc.py [--control-repo <控制仓库根>] [--with-robot]
```

- 自动检查：GPU（nvidia-smi）、左右臂网络（ping 192.168.5.1/2）、串口
  （ttyUSB0/1、ttyACM0/1）、RealSense（lsusb ≥3）、`import dobot_control`、
  jax/GPU、`import openpi_rtc`、相机枚举（serial 与 ini 比对）；
- `--with-robot` 额外做机械臂只读冒烟（需上电）；
- 输出 `robot_pc_check_<时间>.txt` 报告，直接回传即可；
- 全部通过退出码 0，有失败退出码 1；
- **jax / openpi_rtc / cv2 三项必须 OK**（工控机本地推理的前提）。当前系统
  python 3.8.10 没有 jax/cv2/openpi_rtc，必须用 `uv sync` 的 venv 跑
  `check_robot_pc.py`（即上面的 `uv run`），不能用系统 `python3`。

## 阶段 2：推理机（训练/离线评估，GPU 机器）

```bash
nvidia-smi
uv run scripts/benchmark_pi05_inference.py --checkpoint params --num-steps 10 --repeats 20
uv run python measure_latency.py --checkpoint . --hdf5 某帧.hdf5
```

回传：GPU 型号 + mean/median/p95 延迟 + d 推荐表。注意：这里测的延迟只代表
推理机，**实机 d 以工控机（RTX 4090）上的 measure_latency 为准**。

## 阶段 3：checkpoint 摆放与选择

训练产物默认落在（GPU 机器共享文件系统内，路径由 config 名推导）：

```text
checkpoints/qb-ilm-ckpts/g100_pi/<config>/<exp_name>/<step>/
    ├── params/          # 推理需要的权重
    ├── assets/          # norm_stats
    └── train_state/     # 训练状态（推理不需要，可以不拷）
```

**可以放多个、运行时选一个**：`--checkpoint` 接受任意包含 `params/` + `assets/`
的目录。例如：

```bash
--checkpoint "$OPENPI05_CHECKPOINT_49999"                          # baseline（--config pi05-task_00031_yulong-xtrainer）
--checkpoint checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/rtc_train_d7/49999   # train-RTC
--checkpoint checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/pir2_v1/49999        # πR²
```

拷贝到工控机/推理机时只带 `params/` + `assets/` 即可（train_state 可省）。
所有传给 `--checkpoint` / `--dataset` 的路径必须是**绝对路径**，脚本启动时
先校验存在性（checkpoint 需含 `params/`+`assets/`，数据集需含 `.hdf5`），
不满足立即报错，不会进入模型加载。

## 阶段 4：实机验证（工控机 + 机械臂上电后）

### 4.1 镜像测试（防换手，10 秒）

```bash
python3 -c "
import numpy as np
from dobot_control.robots.dobot import DobotRobot
r = DobotRobot('192.168.5.1')
r.command_joint_state(np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0]))  # 只让左臂 J3 微动
"
```

确认动的是**左臂**（数据是 left-first）。如果动的是右臂，说明顺序需要换，立即停。

### 4.2 相机目视比对

三路画面与训练 HDF5 首帧放在一起，确认视角/裁剪一致（top 裁剪
[150:420, 220:480]→640×480）。

### 4.3 工控机本地推理延迟（定 d）

```bash
cd <rtc_control> && uv sync
uv run python measure_latency.py --config pi05-task_00031_yulong-xtrainer \
  --checkpoint . --hdf5 某帧.hdf5 --mode baseline
```

RTX 4090 上 pi0.5 10 步预计 150–300ms；按结果定 d（d=7=280ms 预算，超了就
试 `--num-steps 5` 或增大 d）。

### 4.4 Episode 冒烟（工控机本地跑，不需要网络服务）

```bash
cd <rtc_control>
uv run python run_robot.py --checkpoint . --mode baseline \
  --config pi05-task_00031_yulong-xtrainer \
  --robot robot_xtrainer:XtrainerRobot --episodes 1 \
  [--robot-type "Nova 2"|"Nova 5"]
```

`rtc_control` 解压自推理 bundle（`tar -xzf ... -C <任意目录>`），bundle 已
自包含 openpi 栈 + 控制代码 + ModelTrain + ckpt；`dobot_control` 在
`rtc_control/` 内自动可导入，无需 PYTHONPATH（父目录仅作旧布局回退）。

安全层（`safety.py`）默认开启：有限值、J3 安全位、单步增量 0.9 rad、FK 工作区
/Z 向速度保护（`--robot-type` 必须与实机型号一致）、夹爪裁剪；违规立即中止。
首次运行建议把机械臂速度调低（工控机端 `SpeedFactor`）并有人守在急停旁。

### 4.5 空白 baseline（平台自带 ModelTrain / run_inference.py，对照用）

平台自带一套“加载已有模型实机测试”的链路，可作 baseline 对照。它与我们的
`run_robot.py --mode baseline` 等价，但模型栈不同：ACT（DETR-VAE，torch），
权重是平台训好的 `policy_last.ckpt`。**该栈在工控机原有环境
（Python 3.8.10）跑，不随我们的 bundle 携带**；跑法用平台自己的
`launch_nodes.py` + `run_inference.py`：

```bash
# 0) 用平台自己的 Python 3.8.10（不要用 uv 的 3.11 venv）
cd <工控机手操/控制仓库根>     # 含 ModelTrain、experiments、ckpt 的那套

# 1) 起机器人 ZMQ 服务
python3.8 experiments/launch_nodes.py

# 2) 另一个终端跑原生推理
python3.8 experiments/run_inference.py
```

- 观测/动作约定与我们的实现一致：`{'qpos': 14, 'images': {'top','left_wrist',
  'right_wrist'}}`，输出 14 维（关节弧度 + 夹爪 0~1）；top 相机裁剪
  [150:420,220:480]→640×480。
- 模型：ACT（DETR-VAE），三路 ResNet backbone + transformer decoder，
  chunk=45、temporal aggregation；推理时 latent 置零（不条件于动作）。
  加载 `config.pkl` + `dataset_stats.pkl` + `policy_last.ckpt`。
- 自带安全：关节增量 >0.17 rad 需按键确认、J3/J4 边界（左 J3∈(-2.6,0)、
  J4>-0.6；右 J3∈(0,2.6)、J10<0.6）、FK 工作区（GetPose），违规亮红灯退出。
- 任务切换：`run_inference.py` 按 `put_cube_into_box_yb` → `ckpt_move_cube_new` →
  `clean_dishes` 自动选第一个存在的任务目录，也可用 `XTRAINER_TASK` 指定。
  `Imitate_Model.loadModel()` 会把 ckpt 相对路径拼到 `ModelTrain/` 下，所以
  权重必须放在 `ModelTrain/ckpt/<task>/`。
- Python 版本：ACT 栈用工控机 3.8.10；openpi 栈用 uv 管理的 3.11，两套不混用。
  模型结构见 `ModelTrain/detr/models/detr_vae.py`；调用方式
  `Imitate_Model(ckpt_dir, ckpt_name)` → `loadModel()` → `predict(observation, t)`。
- 对照口径：与我们的 baseline（49999 + `run_robot.py --mode baseline
  --config pi05-task_00031_yulong-xtrainer`）同一场景各 8–10 条，记录
  成功率、碰撞/掉落、边界连续性、视频。

## 阶段 5：对比验收（总预算 ≤4 小时）

5 个条件 × 各 8–10 条：
1. **空白 baseline**：平台原生 ModelTrain（`run_inference.py` +
   `./ckpt/put_cube_into_box_yb/policy_last.ckpt`，bundle 内自带）；
2. **49999 原生 baseline**：`run_robot.py --mode baseline
   --config pi05-task_00031_yulong-xtrainer`（同一 49999 checkpoint）；
3. **inference-RTC（train-free）**：`run_robot.py --mode rtc
   --config pi05-task_00031_yulong-xtrainer`；
4. **train-RTC**：微调 checkpoint + `--config pi05-task_00031_entong-xtrainer`；
5. **πR²**：同上。

记录：成功率（从右架取出并放入左架、未掉落）、碰撞/掉落次数、
关节轨迹边界连续性、视频。同一 prompt：
`Transfer the test tube from the right rack to the left rack.`

注意：5 个条件 × 8–10 条可能超 4 小时预算，可按重要性裁剪
（空白 baseline 可只跑 5 条）或延长预算。
