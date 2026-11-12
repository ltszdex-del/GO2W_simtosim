# 400 轮新策略导入（2026-11-07）

测试环境：`/home/ubuntu/Go2W/interactive_stage1_20261104`；默认地形为滑板公园。

抬腿：`/home/ubuntu/Go2W/go2w_step_sagittal400_20261107/logs/continue400/policy_6699.pt`。
轮驱：`/home/ubuntu/Go2W/go2w_wheel_brake400_20261107/logs/continue400/policy_899.pt`。
两项训练均完成最终评估，exit_code=0。按照用户要求导入最终权重；并非按测试通过率选择最佳 checkpoint。

## 同步内容

- 抬腿参考：vx [-1.5, 2.3]、vy ±1.3 m/s；yaw ±1.2 rad/s；角加速度 1.5、制动 2.4 rad/s²。
- 轮驱参考：vx ±2.5 m/s；yaw ±0.8 rad/s。前进输入加速度 10/3、后退 5、松键制动 4 m/s²。反向按键先与制动叠加至零。转向和平移共享轮速预算。
- `commands.py` 新增由 checkpoint contract 生成参数的方法及轮驱预算约束；`control.py` 在 M 切换时恢复完整命令参数，避免加速度或轮速预算从一个模式泄漏到另一个模式。
- `wheel_adapter.py` 使用实际轮胎碰撞中心在机身坐标系中的 y，执行训练中的 1.5 倍转向前馈；目标加速斜率 60、减速斜率 40 rad/s²；保留官方轮速 30.1 rad/s 与原力矩上限。
- 站立/趴下/起立/倒立/倒立恢复权重保持原配置；启动时共加载七个策略。新增项仅为参数和适配器同步，没有加入新的运动控制器。

## 验证

`logs/dual400_command_verification.json`：每种模式 6000 步随机键盘输入与实际训练 schedule 对照，最大参考误差约 2.1e-6；M 往返后全部命令参数恢复一致。

`logs/dual400_import_check/verified.json`：实际测试机器人在地图中央运行，七个策略全部加载，抬腿→轮驱→抬腿切换；观测当前帧与训练 frame 函数误差低于 6e-7，轮驱腿目标、轮速目标与训练 `_pre_physics_step` 对照误差为零。此检查验证导入接口，不代表地形任务或高速制动通过。

原文件备份：`backups/dual400_20261107/`。包含旧配置、命令生成器、控制器、scene、两套适配器和 main。若回退，应一起恢复这些文件，不能只换旧权重而保留新控制参数。

## 串流

```bash
cd /home/ubuntu/Go2W/interactive_stage1_20261104
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash start_wasd_stream.sh --terrain skatepark
```

WebRTC：106.75.23.13:49100；默认抬腿模式；M 切换轮驱；轮驱支持 W/S/Q/E，抬腿支持 W/A/S/D/Q/E。R 重置到中央 U 站姿。启动日志目录记录在 `logs/current.txt`。

轮驱最终评估通过率约 0.1%，存在高速和转向失稳。抬腿原地 Q/E 最终通过率分别 94.5%/88.3%，组合移动转向约 4.3%；这是测试用最终策略，尚未达到全向转向验收标准。
