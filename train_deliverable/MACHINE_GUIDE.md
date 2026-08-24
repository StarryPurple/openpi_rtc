# 训练机状态指南（给 Agent / 接手人看）

> 读这一份就够继续干活。README_TRAIN.md 讲"怎么装/怎么跑"，这一份讲
> "机器现在是什么状态、踩过什么坑、下一步做什么"。时间：训练开始前排障阶段。

## 1. 机器布局

```text
本地 /root/workspace/embodied-turbo/
├── code/                          ← 训练代码（venv 已装好，code/.venv ~7.9GB）
└── lerobot_cache/                 ← HF_LEROBOT_HOME，转换后的 LeRobot 数据集（video 模式，已就绪）

BOS 挂载 /mnt/bos/keeozw/embodied-turbo/
├── checkpoints/pi05-task_00031_yulong-xtrainer/49999/{params,assets}  ← 微调起点（已校验完整）
├── datasets/task_00031_entong/train/*.hdf5                             ← 原始数据（114 个，转出 112 个 episode）
└── train_out/                    ← 训练产物（本地 code/checkpoints/qb-ilm-ckpts/g100_pi 软链指向这里）

BOS 另有 openpi05/train_code.tar.gz、checkpoint_49999.tar.gz、dataset_entong.tar.gz 三份原始包。
```

## 2. 环境

- Python 3.11（uv 管理），`uv sync --no-dev --no-cache` 装好的 venv（勿再开 cache，磁盘只有 20GB）。
- 关键依赖：`jax[cuda12]==0.5.3`（训练用）、`torch==2.7.1`（仅 lerobot 数据管线用）、
  `av==15.1.0`（pyproject override 固定，14.4.0 wheel 已从 PyPI 下架）。
- 8× ~96GB GPU，JAX 可见 8 卡；NCCL 2.26.2。
- 环境变量（写进 ~/.bashrc 最稳）：
  ```bash
  export BOS=/mnt/bos/keeozw/embodied-turbo
  export OPENPI05_CHECKPOINT_49999=$BOS/checkpoints/pi05-task_00031_yulong-xtrainer/49999
  export OPENPI05_RAW_TRAIN_DIR=$BOS/datasets/task_00031_entong/train
  export HF_LEROBOT_HOME=/root/workspace/embodied-turbo/lerobot_cache
  ```
- norm stats 已就位：`code/assets/pi0-task_00031_entong-xtrainer/task_00031_entong_train/norm_stats.json`
  （注意是 `pi0-` 前缀，加载器只认这个；`pi05-` 那份是旧脚本写的，可留可删）。

## 3. 已确认的事实（不要重复排查）

1. checkpoint 完整：`49999/params/` 内有 `_METADATA`、`manifest.ocdbt`、`ocdbt.process_0/`；
   根目录没有 `_METADATA` 是打包只带 params+assets 所致，正常。
2. 8 卡逐卡压测全过；torch+jax 同进程基础运算全过；单卡训练逻辑尚未验证（见下一步）。
3. 训练卡死/崩溃的根因是 **NCCL 通信**，不是代码、不是 checkpoint、不是数据：
   - 症状：初始化阶段 `CUDA illegal address` / `ncclGroupEnd() failed`；
   - NCCL 把 8 卡当成 8 个节点（`nNodes 8, localRanks 1`），走 **IB/RoCE（mlx5_2）** 通信，
     该 RoCE 路径在本容器不可用；
   - 平台预设了 `NCCL_IB_HCA=mlx5_2`、`NCCL_MIN_NCHANNELS=8`、`NCCL_IB_QPS_PER_CONNECTION=8`
     （多机 IB 训练参数），单机场景需要覆盖。

## 4. 已知坑位与正确姿势（改坏了先看这里）

| 现象 | 原因 | 正确做法 |
| --- | --- | --- |
| 装依赖时 av 编译失败 | av 14.4.0 无 cp311 wheel | pyproject 已 override `av==15.1.0` |
| 磁盘被撑爆 | image 模式转换 + uv cache | 转换必须 `--mode video`；`uv sync --no-dev --no-cache` |
| 转换出"时间戳跳变"坏数据集 | 转换被打断留下半成品 | 删 `$HF_LEROBOT_HOME/task_00031_entong_train` 重转；rtc_train 已加完整性校验 |
| norm stats 找不到 | 写入 `pi05-`、加载器读 `pi0-` | 新版 compute_norm_stats 已修；现机器已手动拷贝 |
| python stdin + DataLoader 报 spawn 错 | multiprocessing 无法重导 stdin 主模块 | 用文件或 `-m` 运行；脚本加 `if __name__ == "__main__":` |
| `_METADATA` FileNotFoundError | weight loader 路径应为 `49999/params` | rtc_train/pir2_train 已修（机器上已 sed） |
| checkpoint 目录已存在 FileExistsError | 上次运行留下目录 | 删 BOS 目录，或新版将支持 `--overwrite` |
| 多卡初始化崩溃/卡死 | NCCL IB/RoCE 不可用 | 见第 5 节 |

## 5. 下一步（按顺序，别跳）

### 5.1 单卡 smoke（验证训练逻辑，绕开 NCCL）
```bash
cd /root/workspace/embodied-turbo/code
CUDA_VISIBLE_DEVICES=0 uv run python /tmp/smoke_1gpu.py 2>&1 | tee /tmp/smoke_1gpu.log
```
期望：出 `Step 0/1`。若崩：贴日志，问题在计算内核/加载路径，与 NCCL 无关。

### 5.2 8 卡 NCCL 修复（强制走 TCP sockets，禁 IB）
```bash
env | grep -i nccl                  # 看平台预设
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=""
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
uv run python /tmp/smoke_nopatch.py 2>&1 | tee /tmp/smoke_ncclib.log
```
仍崩则试：`unset NCCL_SHM_DISABLE`（SHM 开 + P2P 关）；再不行 `NCCL_PROTO=Simple`。
注意：这类 NCCL 修复环境变量必须写进 ~/.bashrc，否则每次新终端失效。

### 5.3 正式训练（8 卡跑通后）
```bash
cd /root/workspace/embodied-turbo/code
uv run python -m openpi_rtc.rtc_train --exp-name rtc_train_d7 \
  --simulated-delay 8 --num-train-steps 10000 --fsdp-devices 2 \
  --save-interval 2500
```
产物在 `$BOS/train_out/g100_pi/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step>/`，
每份含 `params/ + assets/`，可直接拉去工控机测试。

### 5.4 piR2
```bash
uv run python -m openpi_rtc.pir2_train --exp-name pir2_v2 \
  --max-delay 8 --image-delay-max 5 --slow-channel \
  --num-train-steps 10000 --fsdp-devices 2 --save-interval 2500
```

## 6. 提醒

- 长时间训练必须放 tmux / nohup，SSH 断开会杀进程。
- 本地磁盘 20GB：只放 venv + code + 转换数据集；其余一律 BOS。
- 所有 python 运行用文件或 `-m`，别用 stdin heredoc（DataLoader 会挂）。
- 训练侧用 JAX，别切 torch（精度问题）；torch 只是数据管线依赖。
