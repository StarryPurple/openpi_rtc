# openpi_rtc 工控机测试包（test_dobot_rtc_bench）

## 放置图（OBS 传完后的目录安排）

```text
openpi-main/
├── checkpoints/
│   └── dobot/pi05-task_00031_yulong-xtrainer/49999/{params,assets}   ← 49999（单独 OBS 传）
│       （微调产物以后放：dobot/pi05-task_00031_entong-xtrainer/rtc_train_d7/49999/ 等）
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
mkdir -p openpi-main/checkpoints/dobot/pi05-task_00031_yulong-xtrainer
mv inspire/qb-ilm/project/robot-reasoning/xuyue-p-xuyue/ziyu/checkpoints/g100_pi/pi05-task_00031_yulong-xtrainer/default_pi05/49999 \
   openpi-main/checkpoints/dobot/pi05-task_00031_yulong-xtrainer/49999
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

# 4) 微调产物（**必须用修复后代码重训**，见下方修复记录；--checkpoint 覆盖路径）
#    train-RTC 微调产物：
python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc --episodes 5 [--checkpoint <路径>]
#    πR² 微调产物：
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --episodes 5 [--num-steps 10]
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --slow-channel --episodes 5
    # πR² 慢通道 + 单步流（论文 fast mode）：前缀 KV 异步缓存，每
    # --slow-refresh-every(默认5) tick 刷新；每次调用只跑一步 DiT，
    # 释放 d 个干净动作；warmup 按一步 DiT 延迟重新测 d（<=8 训练预算）

# 官方协议对照（pi-r2-flow/pi-r2-flow，2026-09-01 发布）：
#   PI-R2 : --query-mode continuous --chunk-len 2 --nfe 24
#           -> 本 bench: --slow-channel --chunk-len 2（d = slide_steps = chunk_len）
#   RTC   : --query-mode pipelined --chunk-len 5 --nfe 4 --inpaint --force-nonstreaming
#           -> 本 bench: --mode train_rtc（硬冻结前缀 + 非流式完整去噪）
#   plain : --chunk-len 10 --nfe 4（sync）或 continuous + --ensemble（ACT 式时间集成）
#           -> 本 bench: --mode pir2 [--ensemble --chunk-len 10]
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --slow-channel --chunk-len 2 --episodes 5
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --ensemble --chunk-len 10 --episodes 5
```

常用参数：`--episodes`、`--inference-delay`（固定 d，默认自动）、`--execution-horizon`、
`--max-guidance-weight`、`--schedule`、`--arms left|right|both`、`--robot-type "Nova 2"|"Nova 5"`、
`--episode-timeout-s`（默认 30）、`--record-dir`、`--auto-reset`、`--safety-off`（不建议）。

## 输出

```text
openpi-main/records/<模型名>/<mode>/
    episode_N/cam_high.avi
    episode_N/cam_left_wrist.avi
    episode_N/cam_right_wrist.avi
    episode_N.json   # mode/config/checkpoint/d/动作数/时长/延迟统计
```

## 注意事项

- 脚本复用 openpi-main 的 `src/openpi`，并把 yulong/light/entong 的
  TrainConfig 注入 openpi 的 config 注册表（不改 openpi-main 文件）；
  启动自检 `import openpi` 路径不对会直接报错。
- **工控机底层契约（2026-08-26 对方确认，勿回改）**：
  * `DobotRobot.command_joint_state` 收**度数**（内部不再 rad2deg）；
  * `DobotRobot.get_joint_state` 仍返回**弧度**；
  * **夹爪物理映射（2026-08-26 二次实测，曾对调过）**：当前
    `_robot_r`(192.168.5.2) 的夹爪对象驱动【任务】夹爪，
    `_robot_l`(192.168.5.1) 的夹爪对象驱动【未用】夹爪（直连）；方向与
    归一化一致（move(255)=开、move(0)=关，即 1=开、0=关）。若日后夹爪
    又被对调，用 gripper_test.py 重测并按结果改 send_gripper 路由。
  * **不修改 openpi-main 任何文件**：pir2 的逐位置时间通过 action tokens
    注入，adarms_cond 保持逐样本 (B,D)（与 rtc_train 相同契约），gemma
    无需补丁。
  bench 适配：内部/安全检查/位姿对比全部保持**弧度**，仅在发送边界用
  `rad_to_deg` 转度数（夹爪 0~1 不变）；夹爪用 `send_gripper` 按交叉路由
  发送（任务夹爪值 action[13] → `_robot_l` 经 [7:]，未用夹爪值 action[6]
  → `_robot_r` 经 [:7]），不依赖上游 `RealEnv.step_gripper` 的分支。
- **2026-08-25 修复（必须同步更新）**：
  * 根因：`rtc_embed_suffix` / `pir2_embed_suffix` 给 adaRMS 传了
    `(B,H,D)` 逐位置 cond，而 RMSNorm 只支持 `(B,D)` 逐样本，广播成 4 维后
    Attention 的 `q_einsum("BTD,...")` 报
    "Einstein sum subscript 'BTD' does not contain the correct number of
    indices"（train_rtc 工控机实测报错）。训练同样会炸。
  * rtc_train：`rtc_embed_suffix` 已改回逐样本 `(B,D)` cond（Kinetix 语义：
    前缀靠 x_t 钳制 + loss mask）。
  * pir2：逐位置阶梯时间改为注入 action tokens，adarms_cond 保持逐样本
    `(B,D)` —— 全程不修改 openpi-main 的 gemma。
  * 两个采样器补上 `execution_horizon` 参数（bench 会传，旧版会 TypeError）。
  * **旧训练产物作废**：修复改变了训练计算，rtc_train / pir2 必须用
    训练机上的新代码（train_code.tar.gz 含新 rtc_train.py / pir2_train.py /
    gemma.py）重新训练，再回传工控机。
- 安全层默认开（有限值/J3/单步 0.9 rad/FK 工作区），`--robot-type` 必须与实机一致。
- 每集运行中按 **回车** 可提前结束本集（`ended_by="manual"`），否则 30s 超时；
- episode 之间默认人工复位场景（回车继续）；`--auto-reset` 会自行回位（仍建议人守急停）。
- 首次实机先 `--episodes 1`，人守急停。
- probe 的 d 是 p95+1 裕量；train-RTC 需 `d <= simulated_delay-1`，πR² 需 `d <= max_delay`
  （训练默认 8，覆盖 d≤7 / d≤8）。实机 run 模式 d 由实测延迟窗口自动估计。

## 四 mode 全量验证流程（目标：全部跑通）

```bash
cd openpi-main

# ① baseline（49999 原样，先验证控制链路/相机/录像/安全）
python rtc_bench/test_dobot_rtc_bench.py --mode baseline --episodes 3

# ② train-free RTC（同一 49999，推理时引导）
python rtc_bench/test_dobot_rtc_bench.py --mode rtc --episodes 3

# ③ train-RTC（重训产物；训练机 rtc_train_d7 回传后）
python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc --episodes 3 \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step>

# ④ πR²（重训产物；无需修改 openpi-main）
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --episodes 3 \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/pir2_v2/<step>
```

产物都在 `records/<模型名>/<mode>/episode_N/`；每个 mode 跑完核对
`episode_N.json` 的 ended_by（home/timeout/manual）和动作统计。

## 脚本冒烟（未重训前，用已有产物验证代码链路）

train_rtc / pir2 模式可先用**之前训练出的产物**（即使已知是旧版错误条件
下训练的，脚本链路验证不受影响；效果好坏另说）：

```bash
cd openpi-main

# train_rtc 冒烟
python rtc_bench/test_dobot_rtc_bench.py --mode train_rtc --episodes 1 \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/rtc_train_d7/<step>

# pir2 冒烟（不需要改 openpi-main；逐位置时间注入 token、cond 保持 (B,D)）
python rtc_bench/test_dobot_rtc_bench.py --mode pir2 --episodes 1 \
  --checkpoint checkpoints/dobot/pi05-task_00031_entong-xtrainer/pir2_v2/<step>
```

若之前产物尚未放到工控机，可先用 yulong 49999 冒烟（补 `--config
pi05-task_00031_yulong-xtrainer`），pir2 会提示缺少 state_proj/
slow_delay_embed 并随机初始化（WARN 属正常）。
