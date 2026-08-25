# 训练机交付物（train_deliverable）

> 给 Agent / 接手人：先读同目录的 `MACHINE_GUIDE.md`（机器现状、已踩的坑、
> 下一步），再配合本文档使用。

## 结构

```text
train_deliverable/
├── code/                         ← 训练侧代码（独立仓库布局）
│   ├── openpi_rtc/               ← openpi_rtc 包（rtc_train.py / pir2_train.py 等）
│   ├── openpi/  packages/  scripts/  examples/
│   ├── pyproject.toml  uv.lock  .python-version  LICENSE...
├── checkpoints/
│   └── pi05-task_00031_yulong-xtrainer/49999/{params,assets}   ← 微调起点（yulong 49999）
├── datasets/
│   └── task_00031_entong/train/*.hdf5                          ← 微调数据（114 个）
└── README_TRAIN.md
```

## 训练机使用（训练机需可访问 pypi/GitHub 装依赖；无网请另走离线方案）

### 磁盘布局（本地通常只有 ~20GB，务必按此拆）

本地只放 venv + code + 转换后的 LeRobot 数据集（训练热路径，每步随机读帧，
必须本地）。checkpoint（读一次）、原始 hdf5（转换读一次）、训练产物（低频
写入）全部放 BOS 挂载。

```text
本地 /root/workspace/embodied-turbo/
├── code/                      ← 本包解压（uv sync 在此建 .venv，约 5GB）
└── lerobot_cache/             ← 转换后的 LeRobot 数据集（HF_LEROBOT_HOME）

BOS 挂载 /mnt/bos/<user>/embodied-turbo/
├── checkpoints/pi05-task_00031_yulong-xtrainer/49999/   ← 微调起点
├── datasets/task_00031_entong/train/*.hdf5              ← 原始数据
└── train_out/               ← 训练产物（软链指过来，见下）
```

注意：`av` 已通过 pyproject 的 `override-dependencies` 固定为 15.1.0
（14.4.0 的 cp311 wheel 已从 PyPI 下架，源码编译会失败），无需手动处理。

```bash
cd /root/workspace/embodied-turbo/code
uv sync --no-dev --no-cache        # --no-cache：不落 wheel 缓存，省出 4~5GB

# 换成你的 BOS 挂载路径
export BOS=/mnt/bos/<user>/embodied-turbo
export OPENPI05_CHECKPOINT_49999=$BOS/checkpoints/pi05-task_00031_yulong-xtrainer/49999
export OPENPI05_RAW_TRAIN_DIR=$BOS/datasets/task_00031_entong/train
export HF_LEROBOT_HOME=/root/workspace/embodied-turbo/lerobot_cache

# 训练产物写到 BOS（软链零代码改动）
mkdir -p checkpoints/qb-ilm-ckpts
ln -s $BOS/train_out checkpoints/qb-ilm-ckpts/g100_pi

# dry-run 先看调用
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2 --dry-run
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v2 --max-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2 --dry-run

# 正式跑
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v2 --max-delay 8 \
  --image-delay-max 5 --slow-channel --num-train-steps 10000 --fsdp-devices 2
```

πR² v2（慢通道 + 单步流）默认开启 `--slow-channel`：

- 训练时每个样本均匀抽 `d ∈ [1, max_delay]` 作为部署延迟，并另抽
  `k ∈ [0, image_delay_max]` 作为慢通道视觉/语言前缀的“陈旧年龄”；
  数据管线（`SimulateSlowChannel`）会取 `t-k` 帧的 images/state 作前缀，
  当前帧 state 走快速本体通道（`state_proj`），延迟由可学习的
  `slow_delay_embed` 注入动作专家的逐位置 AdaRMS 条件（k=0 时严格为零，
  等价于原模型，不破坏 49999 基础能力）。
- 20% 样本是普通流匹配（共享时间、无 mask、全位置监督），用于推理时
  warm start 流缓冲；阶梯时间加 ±`time_jitter` 对称抖动。
- 推理（工控机 `--mode pir2 --slow-channel`）每次调用只跑一步 DiT：
  缓存前缀 KV 每 `slow_refresh_every` tick 刷新一次，持久化噪声缓冲按
  论文 Fig.2/Eq.4 做一步欧拉 + 滑动，每次释放 `d` 个干净动作。
- 需要的模型侧新参数：`state_proj`（v1 已有）+ `slow_delay_embed`（v2，
  零初始化）。部署 `d <= max_delay`；慢通道预算 `image_delay_max` 要
  >= 实机刷新间隔-1（默认 5 覆盖刷新间隔 5）。

如需 v1（无慢通道、每次调用多步去噪）对比，传 `--no-slow-channel` 即可。

首次运行自动：raw HDF5 → LeRobot 转换（repo id `task_00031_entong_train`）+
norm stats 计算。转换/计算产物在 `code/assets/` 与
`~/.cache/huggingface/lerobot/`（HF_LEROBOT_HOME）下，训练机本地缓存。

## 训练产物与回传

产物默认在：

```text
code/checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/<exp_name>/<step>/
    ├── params/        ← 推理需要
    ├── assets/        ← 含 entong norm_stats
    └── train_state/   ← 可不用
```

回传只带 `params/ + assets/`（按实际保存 step 命名，如 10000/20000/49999），
放到工控机：

```text
openpi-main/checkpoints/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step>/
openpi-main/checkpoints/pi05-task_00031_entong-xtrainer/pir2_v2/<step>/
```

工控机 bench 的 `MODELS` 表里占位 step 是 49999，实际 step 不同时用
`--checkpoint <路径>` 覆盖（或改一行 `MODELS`）。

## 预算提醒

- train-RTC：部署 `d <= simulated_delay - 1`（默认 8 → d≤7）；
- πR²：部署 `d <= max_delay`（默认 8 → d≤8）；
- 实机 probe 测出的 d 超预算时，调大训练预算或减少去噪步数。
