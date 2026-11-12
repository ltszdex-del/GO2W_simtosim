# 测试环境低速站立接口

目录：`/home/ubuntu/Go2W/interactive_stage1_20261104`。只修改测试环境，未修改训练。接口已写入磁盘，当前运行进程不热更新，也不重启串流。

## 配置

```json
{
  "low_speed_hold": {
    "enabled": false,
    "reference_enter_speed_m_s": 0.1,
    "actual_enter_speed_m_s": 0.4,
    "reference_exit_speed_m_s": 0.1
  },
  "policies": {
    "standing_hold": null
  }
}
```

默认关闭，`standing_hold` 为空时也不会停用运动策略。等站立保持策略及其适配器完成后，将 `policies.standing_hold` 填为标准 `{factory, checkpoint, ...}` 配置，通过原有运行时契约校验后再启用。重启测试环境才会加载配置。

进入条件：参考水平合速度 `hypot(vx_ref, vy_ref) < 0.1 m/s`，且实际水平合速度 `hypot(vx, vy) < 0.4 m/s`。两者严格同时满足；等于各自阈值不进入。缺少实际速度或非有限值不进入。当前实际速度来自仿真 root 真值。

退出条件：仅参考水平合速度严格大于 0.1 m/s，恢复 locomotion；实际速度变大不触发退出。等于 0.1 时保持当前模式。站立保持时继续积分 WASD，退出时保留已输入按键和参考速度，不把加速命令清零。恢复运动时重置策略观测历史和相位。

## 与 U 的关系

U 的 `getting_up` 是从趴下起立的一次性动作，不应在低速时从头重放。本接口共用的是 **U 完成后的站立保持策略**：`Controller._enter_standing_hold(state)`。U 完成且站立保持适配器可用时调用它；低速门限满足且启用时也调用它。

当前测试环境 `getting_up` 和 `standing_hold` 槽位都为空，因此未声称已经加载 U 或某个训练完成的站立策略。缺少站立保持适配器时，U 原完成流程继续使用既有站姿 PD 行为。

`standing_hold` 使用与其他 adapter 相同的 `reset/observe/infer/apply` 接口，输出完整 12 腿位置和 4 轮速度目标。只应输出/保持站立目标，不把它标记为一次性完成动作。它需匹配本环境物理、关节顺序、限幅、时序与资产契约；若 U 策略物理定义不同，先实现相应适配，不能仅将路径填入槽位。

该接口不插入 H/P/U 转换中途，也不从 error/prone 等状态自动恢复。原跌倒检测覆盖新增 standing_hold 模式。

## 观测与测试

`/status` 增加 `low_speed_hold` 字段，包含 enabled、ready、requested、reason 和阈值。缺失适配器时 reason 为 `standing_hold_adapter_missing`；默认关闭时为 `disabled`。

```bash
cd /home/ubuntu/Go2W/interactive_stage1_20261104
/home/ubuntu/envs/isaac62/bin/python -m unittest -v test_low_speed_hold test_held_keyboard
```

覆盖合速度而非逐轴判断、边界值、实际速度缺失/非有限值、缺少适配器、默认关闭、进入/退出滞回、WASD 唤醒时保留输入、与 U 共用站立控制、错误/动作转换隔离及原键盘续期回归。
