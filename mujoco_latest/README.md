# Go2W MuJoCo 本地验证

运行 `powershell -ExecutionPolicy Bypass -File "E:/CODEX PROJ/IS/mujoco_latest/start.ps1"`。
本地操作面板：http://127.0.0.1:8776 。原始项目归档未修改训练源码；可编辑测试适配位于 runtime/。

启动脚本使用 `E:/miniconda/envs/mujoco/pythonw.exe`（已验证 MuJoCo 3.3.2）。相机接口同时兼容 3.3.2 的六参数和 3.17 的五参数形式；滚轮、左键拖动均有回归测试 `test_camera.py`。PyCharm 修改文件后需要停止并重新运行程序才能加载修正。

仅保留原 LOW 速度范围，加速度与制动保持原 LOW 规则（即不缩放）：

| 模式 | 前进 / 后退 m/s | 横向 m/s | 静止 / 巡航角速度 rad/s |
|---|---|---|---|
| 抬腿 | 1.15 / -0.75 | 0.65 | 1.57 / 0.8 |
| 轮驱 | 1.5 / -1.25 | 键盘禁用 | 3.14 / 1.0 |

WASDQE 移动；M 切换轮驱/抬腿；H 倒立/回站；P 趴下；U 起立；R 重置。K 保持固定 LOW，不再切到高速。
新版使用独立 GLFW 查看器，没有 MuJoCo Simulate 的内置键盘快捷键。鼠标左拖旋转，滚轮缩放，Esc 关闭。
绿色为参考速度，蓝色为实际速度；竖直箭头表示角速度。界面速度单位为 m/s 和 rad/s。

停车先由移动策略制动。松开移动键、参考线速度和角速度归零（容差 0.001），实际水平速度小于 0.15 m/s 且实际角速度小于 0.15 rad/s，连续保持 0.1 s 后自动进入 `standing_hold`。移动键按下即恢复对应移动策略，不等待额外延时。仍在持续移动时不会自动进入站立。
抬腿/轮驱各三组 W/Q/E 的停车—站立—再启动物理回归通过，见 logs/auto_standing_hold_checks.json。P 在站立保持后还检查实测站姿和关节速度，稳定后才启动卧倒；六次 P/U 往返见 logs/pu_isolation_checks.json。
M 在停稳后切换移动模式并保持站立，收到移动键后恢复目标移动策略。此前 logs/restart_checks.json、logs/mode_switch_checks.json 为自动站立启用前的历史回归。
这只是有限场景回归，尚不等于完整实机或跨引擎验证通过。

策略 50 Hz，外层物理/接触采样 200 Hz；抬腿与轮驱移动的电机/物理更新为 200 Hz，P/U、站立保持和倒立为 1000 Hz，倒立返回过程保留 200 Hz。模型质量 19.523 kg；场地来自原 USD 的 19 个碰撞体，凹台阶/曲面以保持体积的凸棱柱分解。
MuJoCo 与 PhysX 接触和求解不同。MuJoCo 使用速度目标限幅及禁止超速方向继续加力的保护，不通过写关节速度伪造硬速度限制；接触可能导致被动超速，不能宣称实际速度始终满足硬上限。

官方参考：
- https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2w_description/urdf/go2w_description.urdf
- https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py

当前训练和 Isaac 测试采用 30 rad/s 髋/大腿/轮、20.07 rad/s 膝、23.5 N·m 力矩上限。轮目标在 MuJoCo 执行端限制为 30 rad/s。腿部目标变化上限抬腿 8、轮驱 6 rad/s，是目标变化率，不等于实际关节速度。

服务器补训目录 go2w_step_20261111_low_restart 与 go2w_wheel_20261111_low_restart；各 200 轮，2048 并行，从原最新检查点继续。两份补训结果已导入本地：抬腿 `policy_10474.pt`，轮驱 `policy_3824.pt`，存放于 `policies/low_refine_20261111/`。配置直接使用 LOW 训练范围，`command_limit_scale=1`，不再额外减半。重新准备 runtime 会保留这次部署。

早先的独立静态 PD、站姿收拢及通用力矩/阻尼过渡实验已回退。

2026-11-11 P/U 局部修复已完成：P、U、全部站立保持使用 1 ms 电机/求解更新，策略仍为 50 Hz。随后按用户要求，倒立本体也改为 1 ms，H 准备及返回终点不再排除站立防抖；倒立返回动作本身与普通移动仍为 5 ms。P 训练的 5°姿态判定与交互终点分开处理：实际腹部承重、卸力完成、速度稳定且俯仰/横滚小于 8°后，允许进入卧姿并接 U；碰头、关节越限、冲击及严重倾斜保护保留。若下降完成并稳定后腹部离地不足 8 mm，则在 0.5 s 内平滑卸力，让重力完成落地，没有写入身体/关节状态或伪造接触。

轮驱模式中站立保持到收到移动指令，不再下一帧自动交回轮驱。轮驱负责零指令的初始制动，停稳后交给站立。P/U 修复已随新的 10474/3824 移动权重一起验证，倒立及倒立返回策略配置保持原样。

修改前备份：`backups/before_pu_isolation_20261111_191638`。详细原因、验证及范围见 `PU修复说明_20261111.md`。当前回归入口：`check_pu_isolation.py`、`check_pu_scope.py`、`check_stand.py`；`check_pu_handoff.py` 为已回退实验的历史脚本，不代表当前验收流程。

倒立修复已写入正式本地程序，并由 `handstand_edits.json` 在重新准备 runtime 时恢复。备份：`backups/before_handstand_1ms_20261111_205219`。参见 `倒立修复说明_20261111.md` 和物理回归入口 `check_handstand_fix.py`；界面仍按真实时间节奏运行，1 ms 细分不改变仿真时间倍率。
倒立的策略朝向输入以本次进入朝向为基准，加回训练初始朝向，避免移动转向后使用训练范围外的绝对朝向；支撑锚点和返回控制仍使用真实世界朝向。六种进入场景各完成两次倒立、返回、恢复移动验证，未修改策略权重或保护阈值。

2026-11-12 倒立蒸馏已独立启动：当前 teacher 4179 → 历史观测 student，1000 轮、2048 并行。通过服务器配对评估和六种本地连续倒立回归后，后台才会替换 handstand 入口，其他策略及 1 ms/朝向修正保留。当前状态见 `handstand_distill_20261112/background_status.json`；是否已导入以该目录的 `import_result.json` 为准，详细流程见其 `README.md`。完成后需重启测试程序加载新模型。

## 2026-11-12 当前倒立与项目说明

针对性补训学生已完成训练、配对评估和本地六场景回归，正式倒立入口为 `policies/handstand_targeted_20261112/logs/targeted1000/student_999.pt`；轮驱仍固定 3824。上文早期“蒸馏等待中”的描述属于历史过程。

最新说明见 [Sim-to-Sim 项目说明](申校材料/Go2W_Sim2Sim_项目说明.md) 与 [倒立蒸馏问题复盘](申校材料/倒立针对性蒸馏复盘_20261112.md)。本次导入证据为 `handstand_distill_20261112/targeted_import_result.json`，重启 run.py 加载已导入权重。
