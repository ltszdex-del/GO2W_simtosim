# 当前倒立策略蒸馏（2026-11-12）

教师为当前本地实际使用的 `teacher_4179.pt`，SHA256：`6a8978ed66ebd850072a689d6f57694946e818fd812fb835c8a478be30d65eaa`。没有改用未部署的 scratch teacher，也没有修改教师权重、五个接口源文件、参考动作或倒立返回策略。

服务器目录：`/home/ubuntu/Go2W/go2w_handstand_distill4179_20261112`，启动 PID 576，日志 `logs/distill1000/console.log` 和 `progress.jsonl`。DAgger 1000 轮、2048 并行，种子 42，域随机等级沿用教师的 0.2，回合 12 秒。预期输出 `logs/distill1000/student_999.pt`。

采用已有 Student 网络：52 维本体观测 × 50 帧历史 + 57 维目标，共 2657 维，输出 16 维有界动作。教师的 140 维输入含专用特权块，学生输入不包含该块。目标仍包含身体坐标系足端误差和支撑锚点位移，因此这次不等同于完成了仅 IMU/编码器的实机部署。

沿用既有 DAgger，学生访问到的状态由教师提供动作标签。回放从 8192 扩为 131072，前 10% 轮完全使用教师动作，随后线性减少教师执行比例，在 40% 轮处降至零，最后 60% 由学生执行。每轮采样 32 步、监督更新 32 次，Adam 3e-4。动作控制和奖励没有更改；teacher/student 原输入契约通过哈希检查保持一致，蒸馏训练器与入口另存 SHA256。

完成后的验证与导入由 `finish_and_import.py` 在本机隐藏后台运行：

1. 必须完整产生索引 0–999 的进度记录并成功退出。
2. 服务器在相同环境质量参数和初始状态下，对 teacher/student 各做 10 秒评估。student 成功率须至少 90%，且至少达到 teacher 成功率的 90%。
3. 下载并校验导出包、student、教师身份、接口源文件、训练器与入口的 SHA256。
4. 用候选配置运行六种连续物理场景：静止、W 前进、Q 转向、E 转向、E 持续 3 秒、轮驱前进后进入倒立。每种均连续保持 20 秒、返回、再次保持 10 秒、再次返回，再前进和停车，无 reset。
5. 全部通过才写本地 `runtime/config.json` 的 handstand 入口和独立部署记录。失败保留错误日志及原 teacher 配置。

`handstand_student_adapter.py` 继承已经修复的倒立适配器，保留 1 ms 电机积分、50 Hz 策略更新、进入朝向基准、接触/倾斜/关节保护和 H 返回交接。模型及 observe 分支切到 Student，不给动作重复套 tanh。新适配器的权威副本放在本目录，`prepare_runtime.py` 会恢复它；`local_policy_deployment.py` 会恢复验证通过后的 student 部署。

本地接口预检查 `check_student_contract.py` 已通过：2657 维历史构造正确，首次历史填满，向教师特权块注入 NaN 不影响学生观测或动作，1 ms 和朝向修复仍在。该预检查使用临时随机 student 验证接口，不代表训练后的动作质量通过；实际动作质量由完成后的六场景回归判定。

- 后台状态：`background_status.json`；服务器快照：`server_status.json`。
- 启动记录：`background_launch.json`；后台输出：`background_console.log`。
- 成功导入记录：`import_result.json`；各场景输出：`validation_*.log`。
- 模型与训练快照：`../policies/handstand_distill_20261112/`。
- 部署记录：`../policies/handstand_distill_20261112/deployment.json`。
- 原教师连续测试结果仍保留；student 结果写入 `../logs/handstand_student999_*.json`。

本机需保持运行和服务器可连接；后台最长等待六小时，连续连接失败二十次会记录错误并退出。已打开的测试窗口继续使用原内存模型，成功导入后重启 `../run.py` 加载 student。修改前备份为 `../backups/before_handstand_distill_20261112_013349`。抬腿、轮驱、P/U、倒立返回的策略配置不在本次替换范围内。

## 当前实际部署：用户要求启用学生（2026-11-12）

服务器已关机，所有操作使用此前已下载文件。已复核 student_999、教师 4179、导出包及训练源码的 SHA256；学生接口与当前教师完全一致。

本地复查中，学生静止进入倒立后，在倒立控制约 4.88 秒时失稳并触发原失败保护；当前教师通过连续 20 秒、返回、再 10 秒倒立及恢复移动的对照测试。故学生没有通过原先自动部署的六场景稳定性门槛。

用户在获知上述情况后再次明确要求导入。已通过 `activate_requested_student.py` 将 handstand 入口切换到 student_999，同时写入独立部署记录以在 runtime 重建时保留。只执行并通过来源/接口/七策略加载验证，没有把稳定性失败记为通过，也没有放宽保护或改动仿真、控制设置。其他策略均未改动，轮驱仍为提速前的 3824。

正式配置及旧教师入口已备份，位置见 `import_result.json` 的 `config_backup`。当前学生部署带有已知 MuJoCo 失稳限制，重启 `../run.py` 后加载。`background_status.json` 现为 `imported_by_user_request`，不存在新的服务器训练或等待任务。

## 最新状态：针对性补训已通过并导入（2026-11-12）

上述“用户要求启用学生”和失稳限制对应首次蒸馏的旧权重；现已由独立目录 `../policies/handstand_targeted_20261112/logs/targeted1000/student_999.pt` 替换。新权重 SHA256 为 `7301769ab21e881357f4ab1c81e8778b2eed21277a43426c982455af639ac1a7`，完成 1000 次补训更新，并通过服务器独立 20 s 配对评估与 MuJoCo 六场景连续倒立回归。最新部署证据为 `targeted_import_result.json`，状态文件为 `targeted_student_imported`；旧导入记录按历史保留。

详细原因与经验见 [倒立针对性蒸馏复盘](../申校材料/倒立针对性蒸馏复盘_20261112.md)。本次未更换教师、修改物理参数或放宽保护；抬腿及轮驱等其他策略保持不变。
