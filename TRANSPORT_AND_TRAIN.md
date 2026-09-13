# 交付与训练流程（最终版）

两条传输通道，两个目的地：

| 目的地 | 通道 | 桶 | 工具 |
| --- | --- | --- | --- |
| 训练 GPU 机 | 百度 BOS | `handzero-research`（北京） | `bos_transfer.py` |
| 工控机 | 华为 OBS | `handzero-research`（北京四） | `obs_transfer.py` 或 obsutil |

## 一、需要传的文件

打包产物都在本仓库 `upload/` 目录（已打好、已验证）：

```text
upload/
├── train_code.tar.gz         417K  训练侧最新代码 + 传输脚本（code/ + bos_transfer.py 等）
├── checkpoint_49999.tar.gz  ~5-6G  微调起点 49999（yulong 训练，12G 源数据）
├── dataset_entong.tar.gz    2.2G   entong 微调数据（114 个 hdf5）
└── rtc_bench_update.tar.gz  50K    工控机 rtc_bench 更新（test_dobot_rtc_bench.py + openpi_rtc/ 整目录）
```

### A. 训练机（BOS）

上传（需要 `BOS_AK` / `BOS_SK` 环境变量，或改 `bos_transfer.py` 顶部）：

```bash
cd /root/openpi_rtc
python bos_transfer.py upload upload/train_code.tar.gz openpi05/train_code.tar.gz
python bos_transfer.py upload upload/checkpoint_49999.tar.gz openpi05/checkpoint_49999.tar.gz --super-file
python bos_transfer.py upload upload/dataset_entong.tar.gz openpi05/dataset_entong.tar.gz
```

`--super-file` 用 BOS SDK 的 `put_super_object_from_file` 超级文件接口
（多线程并发分片，官方推荐传大文件；12G 的 checkpoint 建议用它）。
分片大小自动对齐 5MB 整数倍，并发数用 `--workers` 调（默认 4）。
注意该接口无断点续传，中断后重跑会重新开始；小文件（<=64MB）自动走
单次 PUT，其余默认走带断点续传的 multipart。

训练机解压（目录结构保持 train_deliverable 布局）：

```bash
mkdir -p train_deliverable && cd train_deliverable
tar -xzf train_code.tar.gz
tar -xzf checkpoint_49999.tar.gz   # -> checkpoints/pi05-task_00031_yulong-xtrainer/49999
tar -xzf dataset_entong.tar.gz     # -> datasets/task_00031_entong
```

### B. 工控机（OBS）

上传（`obs_transfer.py` 默认桶已是 handzero-research / 北京四）：

```bash
cd /root/openpi_rtc
python obs_transfer.py upload upload/rtc_bench_update.tar.gz openpi05/rtc_bench_update.tar.gz
# 或 obsutil：
# obsutil cp upload/rtc_bench_update.tar.gz obs://handzero-research/openpi05/rtc_bench_update.tar.gz -f
```

工控机下载并覆盖：

```bash
cd openpi-main
obsutil cp obs://handzero-research/openpi05/rtc_bench_update.tar.gz . -f   # 或用 download.sh 同款方式
tar -xzf rtc_bench_update.tar.gz        # 覆盖 rtc_bench/（test_dobot_rtc_bench.py + openpi_rtc/）
rm -rf rtc_bench/openpi_rtc/__pycache__ # 清掉旧 pyc，避免版本混乱
```

工控机已有的 49999 checkpoint 放：

```text
openpi-main/checkpoints/dobot/pi05-task_00031_yulong-xtrainer/49999/{params, assets}
```

## 二、训练机（算力平台）操作

### 1. 从 BOS 下载三个包

train_code 包已含 `bos_transfer.py`，先装 BOS SDK 再下载：

```bash
mkdir -p train_deliverable && cd train_deliverable
tar -xzf ../train_code.tar.gz        # 解出 code/ 与 bos_transfer.py 等工具

# 装 BOS SDK（或 pip install -r requirements-bos.txt）
uv pip install -r requirements-bos.txt

export BOS_AK=<你的AK>
export BOS_SK=<你的SK>
python bos_transfer.py download openpi05/train_code.tar.gz       train_code.tar.gz
python bos_transfer.py download openpi05/checkpoint_49999.tar.gz checkpoint_49999.tar.gz
python bos_transfer.py download openpi05/dataset_entong.tar.gz   dataset_entong.tar.gz
```

### 2. 解压与装依赖

```bash
tar -xzf train_code.tar.gz          # 覆盖为最新代码（含工具）
tar -xzf checkpoint_49999.tar.gz    # -> checkpoints/pi05-task_00031_yulong-xtrainer/49999
tar -xzf dataset_entong.tar.gz      # -> datasets/task_00031_entong
cd code
uv sync
```

### 3. 训练两个模型

```bash
export OPENPI05_CHECKPOINT_49999=$(pwd)/../checkpoints/pi05-task_00031_yulong-xtrainer/49999
export OPENPI05_RAW_TRAIN_DIR=$(pwd)/../datasets/task_00031_entong/train

# dry-run 先确认
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2 --dry-run
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v2 --max-delay 8 \
  --image-delay-max 5 --slow-channel --num-train-steps 10000 --fsdp-devices 2 --dry-run

# 正式训练
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 --simulated-delay 8 \
  --num-train-steps 10000 --fsdp-devices 2
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v2 --max-delay 8 \
  --image-delay-max 5 --slow-channel --num-train-steps 10000 --fsdp-devices 2
```

### 4. 回传产物（只带 params + assets）

```bash
cd code
tar -czf ../rtc_train_d7_<step>.tar.gz -C \
  checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step> \
  params assets
tar -czf ../pir2_v2_<step>.tar.gz -C \
  checkpoints/qb-ilm-ckpts/g100_pi/pi05-task_00031_entong-xtrainer/pir2_v2/<step> \
  params assets
cd ..
python bos_transfer.py upload rtc_train_d7_<step>.tar.gz openpi05/rtc_models/rtc_train_d7_<step>.tar.gz
python bos_transfer.py upload pir2_v2_<step>.tar.gz       openpi05/rtc_models/pir2_v2_<step>.tar.gz
```

## 三、回传工控机

开发机从 BOS 拉回模型包，再经 OBS 转给工控机（或直接由工控机从 OBS 拉）：

```bash
# 开发机：BOS 拉回
python bos_transfer.py download openpi05/rtc_models/rtc_train_d7_<step>.tar.gz rtc_train_d7_<step>.tar.gz
python bos_transfer.py download openpi05/rtc_models/pir2_v2_<step>.tar.gz       pir2_v2_<step>.tar.gz
# 开发机：OBS 转给工控机
python obs_transfer.py upload rtc_train_d7_<step>.tar.gz openpi05/rtc_models/rtc_train_d7_<step>.tar.gz
python obs_transfer.py upload pir2_v2_<step>.tar.gz       openpi05/rtc_models/pir2_v2_<step>.tar.gz
```

工控机解压到 `openpi-main/checkpoints/dobot/pi05-task_00031_entong-xtrainer/`
下对应目录（`rtc_train_d7/<step>/`、`pir2_v2/<step>/`）。

## 四、工控机测试

```bash
cd openpi-main

# 0) 自检与探测 d
python rtc_bench/test_dobot_rtc_bench.py --mode probe
python rtc_bench/test_dobot_rtc_bench.py --mode probe --probe-target pir2 --slow-channel

# 1) 对照组
python rtc_bench/test_dobot_rtc_bench.py --mode baseline --episodes 5
python rtc_bench/test_dobot_rtc_bench.py --mode rtc --episodes 5          # train-free RTC（同一个 49999）

# 2) 微调产物（实际 step 不同时用 --checkpoint 覆盖）
python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step>
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --slow-channel \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/pir2_v2/<step>
```

视频输出：`openpi-main/records/<模型名>/<mode>/episode_N/{cam_high,cam_left_wrist,cam_right_wrist}.avi`
+ `episode_N.json`（d / 动作数 / 时长 / 延迟统计）。

## 五、预算与注意事项

- train-RTC：部署 `d <= simulated_delay - 1`（默认 8 → d≤7）。
- πR²：部署 `d <= max_delay`（默认 8 → d≤8）；慢通道 `image_delay_max=5`
  与工控机 `--slow-refresh-every 5` 配套。
- warmup/probe 会按各模式实测延迟自动重测 d；一步 DiT 流若 d>8 会打印警告
  （超训练预算，质量下降）。
- `--mode pir2` 必须用 pir2_v2 微调产物：49999 没有 `state_proj` /
  `slow_delay_embed`，脚本会打印警告（用错产物行为退化）。
- 首次实机先 `--episodes 1`，人守急停；`--auto-reset` 默认开。
- 版本自检：脚本会校验 openpi_rtc 包版本（旧副本缺 `slow_channel` 参数会报错），
  覆盖 rtc_bench 时整个 `openpi_rtc/` 目录一起覆盖并清 `__pycache__`。
