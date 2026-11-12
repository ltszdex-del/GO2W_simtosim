# GO2W 三通道交互环境

更新：2026-11-05，按照《GO2W_环境与PPO训练交接》任务二改造。历史说明见 backups；当前验收记录见 TASK2_ACCEPTANCE.md。

当前机器人由固定站立显式 PD 控制。四个策略槽位仍为 null；键盘改变参考命令，不会让机器人实际行走。H/P/U 明确提示未加载。不能把命令变化或站立稳定当作移动策略验收。

## 输入和状态

| 按键 | 命令 |
|---|---|
| W / S | 前进 / 后退，vx |
| A / D | 左移 / 右移，vy |
| Q / E | 整体左转 / 右转，wz |
| 方向键 | 上下对应 W/S，左右对应 A/D |
| H / P / U | 倒立 / 趴下 / 起立策略请求 |
| F1 | 重新打开串流状态小窗口 |

命令为机身 yaw-only 水平坐标系中的 `[vx, vy, wz]`，单位 m/s、m/s、rad/s。允许组合按键，相反键对该轴制动。按键事件只更新集合，commands.py 每20ms积分。训练端应直接复用此模块，不另写积分器。

默认加速度为 `[0.8,0.4,1.5]`，上限 `[0.6,0.2,0.8]`。平移加速度向量限幅0.8，平移速度投影到椭圆速度域；到顶不积累额外速度。松键使用独立配置的有界制动。转向制动期间继续积分航向目标，角速度归零后锁定。失焦、网页紧急停止和显式释放立即清零参考并撤销输入租约；持键心跳丢失0.4秒触发安全停止。不同来源不能同时获得控制权，过期接管不继承旧速度。

原生 RTC 输入依赖真实客户端事件及 repeat 心跳；服务器虚拟按钮测试不能证明真实客户端键盘通过。网页每100ms发送持键心跳，失焦发送 /release。虚拟按钮短脉冲用于低风险调试，自动释放。

串流面板和网页显示：按键、三轴参考速度、三轴实际加速度、实际速度、航向目标/误差、模式、来源和策略。status schema_version=2，command_axes固定为三轴。实际速度明确来自仿真根状态真值；未来 actor 使用该数据时必须声明可部署的测量/估计接口。

## 运行

服务器目录：`/home/ubuntu/Go2W/interactive_stage1_20261104`。依赖为已有 isaac62 / IsaacLab3，不更改依赖。

```bash
# 先检查共享 GPU 和端口；忙时退出，不停止其他窗口
nvidia-smi
bash ports.sh
# 交互串流，TCP49100 / UDP47998 / loopback桥接8765
bash start.sh --steps 0 --log_dir logs/manual_acceptance
# 或后台启动并记录 logs/current.txt（先检查端口）
bash launch.sh
```

WebRTC 客户端连接 `106.75.23.13`。普通 HTTP GET 返回501仅证明信令服务响应，不能证明媒体流正常。

网页输入隧道，在本地终端保持运行：

```powershell
ssh -p 23 -N -L 8765:127.0.0.1:8765 root@106.75.23.13
```

浏览器打开 `http://127.0.0.1:8765/`。画面在 WebRTC 客户端显示。独立端口入口 start_indie.sh 使用 TCP49101 / UDP48000 / 桥接8766，须使用支持对应端口的客户端；它不会解决共享 GPU 显存不足的问题。

```bash
# 无 GPU 逻辑测试
/home/ubuntu/envs/isaac62/bin/python -m unittest -v test_control test_keyboard_router test_stream_ui test_task2
# 有限时、无串流、真实物理测试；不启动训练
bash run_task2_verification.sh
```

## 物理与适配器

物理5ms、策略20ms，每次物理子步重算显式力矩。腿kp=70/kd=2.5，轮kd=0.2，力矩±23.5Nm；地面摩擦0.8。初始化后不重写根姿态/速度。输出为按FR/FL/RR/RL顺序的12腿位置rad和4轮速rad/s，经过关节限位、目标变化率及力矩约束。environment_manifest.json记录实际资产哈希、关节顺序、输出定义、PD和时序。

策略配置必须是对象，包含 factory=`module:factory` 和 checkpoint 路径；工厂接收 `(spec, device)` 返回适配器。裸权重路径会被拒绝。适配器实现 load/reset/observe/infer/apply/is_complete，apply返回 leg_pos_rad 和 wheel_vel_rad_s。模块负责根据训练定义构造观察、归一化、历史和完成判据，不能只返回推理结果而跳过真实电机目标。

manifest必填观察顺序、normalization、scales、history、phase、previous_action、target、task_id、joint_order、output、pd、limits、timing、asset_sha256；新P/U应显式 phase:none。加载前校验实际环境的顺序、输出、PD、限幅、时序和哈希。缺失/不兼容策略会记录 policy_errors 并保持站姿。观察等策略专用字段由适配器按照权重定义解释；环境不会猜测模型接口。

同一时刻只执行一个适配器。P/H请求先制动，参考和真实速度均稳定0.3秒后才切换；执行时运动命令关闭。P完成保持当前实际目标，U完成平滑回站姿。动作完成、危险接触及任务特定失败判据由相应适配器负责；通用站姿倒地检查只适用于standing/locomotion，不将正常趴姿或倒立判为站姿摔倒。

本轮仅证明环境和适配器执行通路；真实移动、P/U/H能力需要匹配权重后单独验收。低速航向误差、原地转漂移、跟踪RMSE和侧滑尚未评估。
