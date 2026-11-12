# U/P/WASD 集成与倒立续训

交互项目：`/home/ubuntu/Go2W/interactive_stage1_20261104`。

## 已导入权重

| 用途 | 权重绝对路径 |
|---|---|
| U 起立、无速度输入的站立保持 | `/home/ubuntu/Go2W/pu_ppo_u_vertical100_20261106/logs/U_vertical_resume100/final.pt`，内部 iteration=1099 |
| P 趴下 | `/home/ubuntu/Go2W/pu_ppo_p_fromU1050_200_20261106/logs/P_fromU1050_scratch200/final.pt`，内部 iteration=199 |
| WASD | `/home/ubuntu/Go2W/go2w_wasd_directional_resume300_20261106/logs/continue300/policy_3399.pt` |

`config.json` 是当前运行配置。每次重启都会严格校验权重对应的训练源文件与资产哈希。PU 源程序分别复制到 `pu_u_runtime`、`pu_p_runtime`，仅改写同目录模块导入，并用 `pu_bridge.py` 将构造器绑定到现有机器人。每份副本的 `runtime_transform_manifest.json` 记录变换前后的哈希。原始训练项目不修改。

U 使用216维、P使用279维、WASD使用570维观测；各自保留原始历史、动作解释和控制逻辑。U/P保留支撑前馈、姿态补偿及U额外髋关节刚度，不以普通位置目标PD替代完整控制器。

## 运行时切换

- 初始姿态来自最终 U 的实测站姿，初始电机目标采用同次采样的承载目标；初始化完成后，切换策略不写入根姿态或关节状态。
- 水平参考速度模长 `<0.1 m/s` 且实际速度模长 `<0.4 m/s` 时进入 U 保持；只有参考速度 `>0.1 m/s` 才切回 WASD。
- WASD 的四轮隐式阻尼100；U/P/H使用0并交给原训练显式力矩控制器。物理步5ms，控制步20ms，实际执行4次力矩更新。
- P/H请求先刹停。U保持状态下，切换前要求参考速度近零、水平实际速度<0.10m/s、偏航角速度<0.35rad/s，持续0.3s。H另核对起始关节姿态和目标速度。
- U→保持沿用同一U控制上下文，不重播起立。训练原始成功标志和交互切换标准分开记录：交互标准检查高度、姿态、对称性、轮轴倾斜、速度与0.5s姿态变化，并再持续0.5s，不沿用训练的起立截止时间；所有原始失败保护仍有效。
- P完整训练成功后进入prone。如果P尚未达到训练成功标准，但已接近趴姿、低速、水平且无失败保护触发，允许用户按U起立。此时记录为物理姿态就绪，不记作P训练成功。
- 当前WASD无QE转向能力，维持原有禁用行为。

## 资产与风险

交互环境使用U/P资产：`/home/ubuntu/Go2W/pu_ppo_u_vertical100_20261106/assets/go2w_description`。它与旧WASD/H资产仅在base、Head_upper、Head_lower碰撞标签不同；`asset_compatibility.py` 删除这三个部位碰撞标签后，对剩余URDF逐项比对，拒绝其它差异。WASD还核验其余资产字节。此迁移会改变机身触地行为，应继续验收各方向运动和倒立。

P末轮训练日志的严格成功率为0。实际回放可到约0.095m趴姿，无失败保护触发，但电机增益仍为1，未完成训练要求的支撑确认、卸载断力和持续保持；不能宣称P已训练达标。

U可达到约0.391m站姿，但原训练成功标志也尚未稳定满足，存在高频速度/力矩振荡。交互切换成功不等于训练稳定性达标。`status.json.physics.pu`保留原始成功、motor_gain、角速度、关节误差与原始hold，方便后续排查。

实测回放：`logs/pu_integration3`，顺序为U保持→W→刹停回U保持→P→物理趴姿就绪后U→U保持。该测试只验证这一条指令序列，不能替代多初态、多方向成功率评估。

## 倒立300轮

新项目：`/home/ubuntu/Go2W/go2w_handstand_fromU300_20261106`。

从旧`teacher_3799.pt`保留网络、优化器及课程状态，续训300轮、1024环境，目标4099。初始q、轮角、根位置和姿态来自`U_final_standing.json`的最终U实测站姿；根位置XY归零，Z加2mm避免初始穿地。参考动作前0.6s通过最小加加速度多项式平滑衔接，重算关节速度和足端FK；原奖励与控制参数保持。

训练日志：`/home/ubuntu/Go2W/go2w_handstand_fromU300_20261106/logs/train300/console.log`。

```bash
tail -f /home/ubuntu/Go2W/go2w_handstand_fromU300_20261106/logs/train300/console.log
```

最终权重预计为`/home/ubuntu/Go2W/go2w_handstand_fromU300_20261106/logs/train300/teacher_4099.pt`。完成训练不自动代表通过倒立验收；H线上权重以`config.json`为准。

## 启动与回退

```bash
bash /home/ubuntu/Go2W/interactive_stage1_20261104/start_wasd_stream.sh
```

主串流日志由`logs/current.txt`指向。新集成前备份：`backups/before_pu_20261106T131402Z`；回退需整体恢复同批源文件和配置，避免混用控制器和URDF。

## 最终验收更新

300轮倒立续训已完成，最终4099已通过700步/14s短回放，并写入H配置。此为短时功能检查，不是严格稳定性成功率达标。详见`integration_report_20261106.json`和`logs/h_fromU_final/handstand_test.json`。
{
  "iterations": 300,
  "console_last_iteration": 4099,
  "checkpoint_iteration": 4099,
  "last_success": 0.0,
  "last100_episode_weighted_success": 0.0004206730796655533,
  "last_hold_s": 0.7416420598183909
}
