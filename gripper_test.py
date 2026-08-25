#!/usr/bin/env python3
"""夹爪物理映射探测（交互式，无需区分左右）。

只区分两个物理夹爪：
  * 任务夹爪：用来夹试管的那只
  * 未用夹爪：另一只（任务中保持张开）

运行后在 openpi-main 根目录:
    python gripper_test.py
按提示回答即可，最后把终端输出贴回开发。
"""

from __future__ import annotations

import time

from examples.xtrainer_real.robots.dobot import DobotRobot

TASK_IP = "192.168.5.2"    # 任务臂（bench 默认 arms=right 用的就是它）
OTHER_IP = "192.168.5.1"   # 另一臂


def ask(prompt: str, choices: tuple[str, ...]) -> str:
    hint = "/".join(choices)
    while True:
        ans = input(f"{prompt} [{hint}]: ").strip()
        if ans:
            return ans


def test(ip: str, label: str) -> None:
    print(f"\n===== {label}（{ip}）=====")
    bot = DobotRobot(ip, no_gripper=False)
    print(
        f"配置: port={bot.com_list[ip]} id={bot.id_list[ip]} "
        f"servo_pos={bot.servo_pos_list[ip]}"
    )
    for pos, pname in ((0, "0(理论=开)"), (255, "255(理论=关)")):
        print(f"\n>>> 现在命令【{label}】的夹爪对象 move({pname})")
        bot.gripper.move(pos, 100, 1)
        time.sleep(1.5)
        moved = ask("   哪个物理夹爪动了？", ("任务夹爪", "未用夹爪", "都没动", "都在动"))
        direction = ask("   它是在张开还是闭合？", ("开", "关", "看不清"))
        try:
            cur = bot.gripper.get_current_position()
            print(f"   读回 get_current_position = {cur}")
        except Exception as e:  # noqa: BLE001
            print(f"   读回失败: {type(e).__name__}: {e}")
        print(f"   记录: 动了={moved}, 方向={direction}")


def main() -> int:
    print("准备：确认机械臂周围没有障碍物，夹爪不会夹到东西。")
    input("按回车开始（会真实开关夹爪数次）...")
    test(TASK_IP, "任务臂夹爪对象")
    test(OTHER_IP, "另一臂夹爪对象")
    print("\n完成。请把以上输出贴回开发。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
