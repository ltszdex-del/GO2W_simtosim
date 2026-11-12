# 轮驱速度与移动转向补训（2026-11-12）

服务器任务：`/home/ubuntu/Go2W/go2w_wheel_speed250_turn150_20261112`，训练 PID 246。
从 LOW 第 3824 轮续训 200 轮，4096 并行，完成检查点应为 `policy_4024.pt`。保留模型、优化器、随机数状态、左右镜像、组合转向采样、站姿/平衡/接触奖励和官方电机限制。

前进上限 1.5 → 2.5 m/s；满速转向上限 1.0 → 1.5 rad/s。原地上限 3.14 rad/s、速度相关指数衰减系数 3、后退 -1.25 m/s、输入/制动加速度保持不变。训练中的旧硬编码课程上限和对应日志改为读取当前 COMMAND，避免记录与实际配置不一致。

本地已启动独立后台进程 `finish_and_import.py`。每 45 s 检查一次训练，成功退出且完整产生 200 条连续更新后，下载服务器导出包、校验 SHA256 和训练源文件、加载全部七个策略入口并执行轮驱启动/停车物理检查，然后导入新轮驱权重与匹配的限速配置。若训练、下载或验证失败则记录错误，不部署。

- 当前后台状态：`background_status.json`；训练状态快照：`server_status.json`。
- 后台启动记录：`background_launch.json`；输出：`background_console.log`。
- 导入完成证明：`import_result.json`；验证输出：`import_console.log`。
- 完成后策略根目录：`../policies/wheel_speed250_20261112/`。
- 新轮驱部署记录：`../policies/wheel_speed250_20261112/deployment.json`，由 `local_policy_deployment.py` 在 runtime 重建时恢复。

测试程序已经运行时不会强行替换内存中的模型；导入完成后重新启动 `../run.py` 加载。后台进程需要本机保持运行和服务器可连接，最长等待四小时，连续连接失败二十次后保留错误记录。

修改前本地备份：`../backups/before_wheel_speed250_20261112_012259`。抬腿、P/U、倒立和倒立返回的配置不在本次部署范围内。
