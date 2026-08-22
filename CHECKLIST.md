# 环境检查与验收清单（工控机 / 推理机 / GPU 机器）

机器分工（已确认）：

- **推理机（GPU 机器）**：训练 + 离线评估 + probe。与当前开发机共享文件系统。
- **工控机**：实机执行——**自己加载已经训练好的 checkpoint，实时跑模型推理
  并驱动机器人**（RTX 4090 / 24GB，Python 3.13.5）。需要装好 openpi + jax
  (CUDA) + cv2 + pyrealsense2 + dobot_control；不需要任何训练功能。
- 三方法（inference-RTC / train-RTC / πR²）的模型都是训练好的 checkpoint，
  实机对比时在工控机上逐个加载。

---

## 阶段 0：GPU 机器（训练/评估）

```bash
bash openpi_rtc/run_probe.sh        # 需要 OPENPI05_RAW_TRAIN_DIR（或第二个参数）；输出 probe_report_*.txt
```

通过后：

```bash
uv run python openpi_rtc/eval_offline_rtc.py --mode baseline \
  --checkpoint "$OPENPI05_CHECKPOINT_49999" \
  --dataset "$OPENPI05_RAW_TRAIN_DIR"
```

## 阶段 1：工控机环境检查（一条命令）

```bash
# 在工控机上运行（dobot_control 仓库根目录，或加 --control-repo 指定）
python3 openpi_rtc/check_robot_pc.py [--control-repo <控制仓库根>] [--with-robot]
```

- 自动检查：GPU（nvidia-smi）、左右臂网络（ping 192.168.5.1/2）、串口
  （ttyUSB0/1、ttyACM0/1）、RealSense（lsusb ≥3）、`import dobot_control`、
  jax/GPU、`import openpi_rtc`、相机枚举（serial 与 ini 比对）；
- `--with-robot` 额外做机械臂只读冒烟（需上电）；
- 输出 `robot_pc_check_<时间>.txt` 报告，直接回传即可；
- 全部通过退出码 0，有失败退出码 1；
- **jax / openpi_rtc / cv2 三项必须 OK**（工控机本地推理的前提）。当前系统
  python 3.13.5 缺这些时，用与训练机一致的 Python 3.11 建 venv 再装。

## 阶段 2：推理机（训练/离线评估，GPU 机器）

```bash
nvidia-smi
uv run scripts/benchmark_pi05_inference.py --checkpoint params --num-steps 10 --repeats 20
uv run python openpi_rtc/measure_latency.py --checkpoint . --hdf5 某帧.hdf5
```

回传：GPU 型号 + mean/median/p95 延迟 + d 推荐表。注意：这里测的延迟只代表
推理机，**实机 d 以工控机（RTX 4090）上的 measure_latency 为准**。

## 阶段 3：checkpoint 摆放与选择

训练产物默认落在（GPU 机器共享文件系统内）：

```text
checkpoints/pi05-task_00031_yulong-xtrainer/<exp_name>/<step>/
    ├── params/          # 推理需要的权重
    ├── assets/          # norm_stats
    └── train_state/     # 训练状态（推理不需要，可以不拷）
```

**可以放多个、运行时选一个**：`--checkpoint` 接受任意包含 `params/` + `assets/`
的目录。例如：

```bash
--checkpoint checkpoints/pi05-task_00031_yulong-xtrainer/49999                    # baseline
--checkpoint checkpoints/pi05-task_00031_yulong-xtrainer/rtc_train_d7/49999      # train-RTC
--checkpoint checkpoints/pi05-task_00031_yulong-xtrainer/pir2_v1/49999           # πR²
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
uv run python openpi_rtc/measure_latency.py --checkpoint <模型目录> --hdf5 某帧.hdf5
```

RTX 4090 上 pi0.5 10 步预计 150–300ms；按结果定 d（d=7=280ms 预算，超了就
试 `--num-steps 5` 或增大 d）。

### 4.4 Episode 冒烟（工控机本地跑，不需要网络服务）

```bash
export PYTHONPATH=/path/to/openpi05:$PYTHONPATH
python3 openpi_rtc/run_robot.py --checkpoint <模型目录> --mode baseline \
  --robot openpi_rtc.robot_xtrainer:XtrainerRobot --episodes 1 \
  [--robot-type "Nova 2"|"Nova 5"]
```

安全层（`openpi_rtc/safety.py`）默认开启：有限值、J3 安全位、单步增量
0.9 rad、FK 工作区/Z 向速度保护（`--robot-type` 必须与实机型号一致）、
夹爪裁剪；违规立即中止。首次运行建议把机械臂速度调低（工控机端
`SpeedFactor`）并有人守在急停旁。

## 阶段 5：对比验收（总预算 ≤4 小时）

4 个模型 × 各 8–10 条：baseline / inference-RTC / train-RTC / πR²。
记录：成功率（从右架取出并放入左架、未掉落）、碰撞/掉落次数、
关节轨迹边界连续性、视频。同一 prompt：
`Transfer the test tube from the right rack to the left rack.`
