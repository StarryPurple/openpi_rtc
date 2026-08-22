# openpi_rtc 工控机测试包（test_dobot_rtc_bench）

## 放置图（OBS 传完后的目录安排）

```text
openpi-main/
├── checkpoints/
│   └── pi05-task_00031_yulong-xtrainer/49999/{params,assets}   ← 49999（单独 OBS 传）
│       （微调产物以后放：pi05-task_00031_entong-xtrainer/rtc_train_d7/49999/ 等）
├── records/                          ← 运行时自动创建
│   └── <模型名>/<mode>/episode_N/*.avi + episode_N.json
└── rtc_bench/                        ← 本包 rtc_bench/ 目录整体拷到这里
    ├── test_dobot_rtc_bench.py       ← 唯一入口（在 rtc_bench 根）
    ├── check_robot_pc.py             ← 环境自检（在 rtc_bench 根）
    └── openpi_rtc/                   ← 真正的包（__init__.py 等 11 个模块）
        （不含 openpi！运行时复用 openpi-main/src/openpi）
```

本包含三部分：
- `rtc_bench/`：代码（不含 openpi 副本——运行时复用 openpi-main 的
  `src/openpi`，脚本会把 yulong / light / entong 三个 task config 注入
  openpi 的 config 注册表，不改 openpi-main 文件）；
- `yulong.tar.gz`：49999 模型（42GB，含 train_state；推理只需
  params+assets，train_state 可删）；
- `README_RTC_BENCH.md`：本说明。

> 如果之前已传过旧版（带 openpi/ 的），请手动清理后替换脚本：
> ```bash
> # 旧版 rtc_bench/*.py 在根目录、新版在 rtc_bench/openpi_rtc/，
> # 最简单是直接重传整个 rtc_bench/（160K）覆盖
> rm -rf openpi-main/rtc_bench
> # 然后把新 rtc_bench/ 放到 openpi-main/ 下
> ```

## 49999（已在包内：yulong.tar.gz）

解压注意：这个 tar 的条目带长路径前缀（`inspire/qb-ilm/.../49999/`），
解压后需要移动到位：

```bash
tar -xzf yulong.tar.gz     # 生成 inspire/qb-ilm/.../default_pi05/49999/
mkdir -p openpi-main/checkpoints/pi05-task_00031_yulong-xtrainer
mv inspire/qb-ilm/project/robot-reasoning/xuyue-p-xuyue/ziyu/checkpoints/g100_pi/pi05-task_00031_yulong-xtrainer/default_pi05/49999 \
   openpi-main/checkpoints/pi05-task_00031_yulong-xtrainer/49999
rm -rf inspire            # 清理残留路径；train_state 推理不需要，可一并删
```

（嫌 42GB 大或不想搬路径，我可以重打一个 ~12GB 的 params+assets 精简包，
顶层直接是 `pi05-task_00031_yulong-xtrainer/49999/`，解压即到位。）

## 代码包传输（OBS）

开发机上传：

```bash
obsutil cp <deliverable目录> obs://handzero-research/openpi05/rtc_bench_deliverable -r -f
```

工控机下载并摆放（最终结构见顶部放置图）：

```bash
obsutil cp obs://handzero-research/openpi05/rtc_bench_deliverable ./ -r -f
mv <deliverable>/rtc_bench openpi-main/rtc_bench
# 模型按上面的 yulong.tar.gz 解压说明处理
```

## 运行（openpi-main 根目录，用 openpi-main 的 venv python 3.11）

```bash
cd openpi-main

# 0) 自检：确认复用 openpi-main 的 src/openpi（而不是残留的 rtc_bench/openpi）
python -c "import openpi; print(openpi.__file__)"   # 应显示 .../openpi-main/src/openpi/__init__.py

# 1) 探测延迟，推荐 d（真机取观测；或加 --hdf5 某帧.hdf5 不用真机）
python rtc_bench/test_dobot_rtc_bench.py --mode probe
python rtc_bench/test_dobot_rtc_bench.py --mode probe --probe-target rtc

# 2) 普通 baseline（49999 原样）
python rtc_bench/test_dobot_rtc_bench.py --mode baseline --episodes 5

# 3) train-free RTC（同一个 49999，推理时引导；d 自动估计）
python rtc_bench/test_dobot_rtc_bench.py --mode rtc --episodes 5

# 4) 微调产物（训练后再跑；--checkpoint 可覆盖路径）
python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc --episodes 5 [--checkpoint <路径>]
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --episodes 5 [--num-steps 10]
```

常用参数：`--episodes`、`--inference-delay`（固定 d，默认自动）、`--execution-horizon`、
`--max-guidance-weight`、`--schedule`、`--arms left|right|both`、`--robot-type "Nova 2"|"Nova 5"`、
`--episode-timeout-s`（默认 60）、`--record-dir`、`--auto-reset`、`--safety-off`（不建议）。

## 输出

```text
openpi-main/records/<模型名>/<mode>/
    episode_N/cam_high.avi
    episode_N/cam_left_wrist.avi
    episode_N/cam_right_wrist.avi
    episode_N.json   # mode/config/checkpoint/d/动作数/时长/延迟统计
```

## 注意事项

- 脚本强制使用 `rtc_bench/openpi`（vendored，含 task 注册），不碰 openpi-main 的
  `src/openpi`；启动自检 `import openpi` 路径不对会直接报错。
- 安全层默认开（有限值/J3/单步 0.9 rad/FK 工作区），`--robot-type` 必须与实机一致。
- episode 之间默认人工复位场景（回车继续）；`--auto-reset` 会自行回位（仍建议人守急停）。
- 首次实机先 `--episodes 1`，人守急停。
- probe 的 d 是 p95+1 裕量；train-RTC 需 `d <= simulated_delay-1`，πR² 需 `d <= max_delay`
  （训练默认 8，覆盖 d≤7 / d≤8）。实机 run 模式 d 由实测延迟窗口自动估计。
