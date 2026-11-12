# H 倒立接入记录

测试环境：`/home/ubuntu/Go2W/interactive_stage1_20261104`。

H 使用 `/home/ubuntu/Go2W/Go2WMimicHandstand_v2/logs/train/2026-11-05_hold_joint_speed_resume300/teacher_3799.pt`，以及同目录 `params/code`、`params/stand_to_handstand.npz`。按原 `USER_DECISION.json`，3999 曾被用户否决，3799 为保留基准。

WASD 仍使用 flexible_gait 项目的 2799；P/U/standing_hold 保持未加载，低速站立接口仍关闭。训练文件和权重未修改。

## 切换与控制

- H 从 standing/locomotion/standing_hold 请求切换；已在倒立中再次按 H 不重置参考时钟。
- 先保持 WASD 策略，撤去移动加速度并按原规则制动。参考速度接近零且实际平移速度低于 0.15 m/s 后，退出运动策略，用原 PD 跟踪倒立参考首帧关节目标。
- 参考及实际三个速度分量均低于 0.05，目标已到达、关节已稳定，连续满足 0.3 秒后，启动 H。
- H 运行时将实际轮阻尼从 100 改为 0，并验证写入结果；使用原始 0.2 轮速环、70/2.5 腿 PD、目标速度前馈、8 rad/s 目标限速和 23.5 Nm 力矩限幅。
- 物理步 5 ms，每控制周期实际重算 4 次原始 PD。H 的控制与观测直接绑定经过 checkpoint 哈希验证的原始 MimicEnv 方法，不重新编写近似公式。
- 140 维教师观测、16 动作、参考时钟、上一动作及历史在 H 进入时独立初始化。支撑锚点及任务水平原点取进入 H 时的位置，未向仿真写入任何位姿。
- 参考末帧继续闭环保持。沿用原碰撞、前轮平行度、姿态、非轮接触及高度失败判据。失败进入 error 并停止策略更新，不自动重置或重播；重新测试需重启环境。

## 已验证及限制

21 项控制切换、低速接口与键盘回归通过。实际 500 步检查记录在 `logs/handstand_integration_smoke4`：H 控制运行 5.54 秒，峰值 pitch 1.528 rad（约 87.6°），pitch > 1.3 rad 的累计时间 2.36 秒；随后因低于参考高度超过 0.12 秒触发原失败保护，未发生轮碰撞。该检查验证可进入倒立及失败保护，**不代表稳定保持验收通过**。

原始完整 MimicEnv、同权重、单环境、无随机化对照也在第 268 步（约 5.38 秒）失败，日志为 `logs/handstand_baseline.log`。因此这次接入保留了原策略已知的保持不稳定性，未通过关闭保护掩盖失败。

启动：

```bash
bash /home/ubuntu/Go2W/interactive_stage1_20261104/start_wasd_stream.sh
```

连接原串流客户端，点击视口后按 H，或点击右侧 H 按钮。切换等待阶段可在状态中看到 `pending=handstand`。日志 `[MODE] handstand` 标记解除轮锁及开始倒立；`[HANDSTAND_FAILURE]` 记录具体失败指标。

改动前文件在 `backups/before_handstand_*`，最初接入前版本为 `before_handstand_20261106T101545Z`。
