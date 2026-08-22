# 训练机交付物（train_deliverable）

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

```bash
cd train_deliverable/code
uv sync

export OPENPI05_CHECKPOINT_49999=$(pwd)/../checkpoints/pi05-task_00031_yulong-xtrainer/49999
export OPENPI05_RAW_TRAIN_DIR=$(pwd)/../datasets/task_00031_entong/train

# dry-run 先看调用
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2 --dry-run
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v1 --max-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2 --dry-run

# 正式跑
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v1 --max-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2
```

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
openpi-main/checkpoints/pi05-task_00031_entong-xtrainer/pir2_v1/<step>/
```

工控机 bench 的 `MODELS` 表里占位 step 是 49999，实际 step 不同时用
`--checkpoint <路径>` 覆盖（或改一行 `MODELS`）。

## 预算提醒

- train-RTC：部署 `d <= simulated_delay - 1`（默认 8 → d≤7）；
- πR²：部署 `d <= max_delay`（默认 8 → d≤8）；
- 实机 probe 测出的 d 超预算时，调大训练预算或减少去噪步数。
