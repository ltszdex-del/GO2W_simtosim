# 复现与版本说明

整理日期：2026-11-12。本仓库为当前可运行的 MuJoCo 项目及相关训练资料的发布副本。原始工作区包含更多试验、备份和运维文件，这些没有全部上传。

## 本地运行

使用根目录 `requirements.txt` 安装 CPU 推理依赖，运行 `python mujoco_latest/run.py`。当前已验证环境为 Windows、CPython 3.9、MuJoCo 3.3.2、NumPy 2.0.2、PyTorch 2.8.0+cpu、GLFW 2.9.0；GPU 并非必要。页面控制端口为 8776，默认只监听本机。

程序包含预生成的 `mujoco_latest/generated/scene.xml`、碰撞网格、Unitree 渲染网格与部署权重，因此常规运行不依赖 OpenUSD、SciPy 或 Isaac Sim。保留目录结构，包括名称以点开头的 `.tmp_unitree_mujoco`。

若要从 URDF / USD 重新构建模型，额外安装 `requirements-build.txt`，再运行：

```powershell
python mujoco_latest/build_scene.py
```

构建会生成新的模型文件；如 MuJoCo、USD 或几何库版本发生变化，应重新检查质量、惯量、碰撞几何与动作回归，不能仅凭构建成功认定控制行为一致。

## 当前策略身份

完整路径和 SHA256 保存在 [POLICY_MANIFEST.json](POLICY_MANIFEST.json)。可运行以下命令核对：

```powershell
python tools/verify_snapshot.py
```

| 入口 | 版本 / 身份 |
|---|---|
| locomotion | 抬腿 10474 |
| wheel_locomotion | 轮驱 3824，提速前基线 |
| handstand | 针对性蒸馏学生 999，SHA256 `7301769ab21e881357f4ab1c81e8778b2eed21277a43426c982455af639ac1a7` |
| handstand_return | 返回 0349，保留其原教师依赖 |
| lying_down | P final，训练索引 199 |
| getting_up / standing_hold | U final，训练索引 1099，同一份权重 |

原倒立教师为 4179，SHA256 `6a8978ed66ebd850072a689d6f57694946e818fd812fb835c8a478be30d65eaa`，继续作为适配合同、参考动作和诊断依赖保留。首次蒸馏也有同名 `student_999.pt`，必须按完整目录和哈希区分。

## 源码与路径约定

- `mujoco_latest/runtime/` 是当前运行适配代码；`prepare_runtime.py` 可以按原快照与本地补丁重建该目录。
- `Go2W_项目整理_20261111/server_snapshot/` 是日期明确的源码、资产及依赖快照，其哈希约束源文件保持原始字节。`.gitattributes` 关闭了自动换行转换，避免 Windows checkout 导致合同不匹配。
- 活动配置和两个部署记录中的工作区绝对路径改为 `__GO2W_WORKSPACE__/`，由 `bootstrap.py` 在运行时解析；历史 `/home/ubuntu/Go2W/` 映射到仓库内的 `server_snapshot/`。
- 生成 MJCF 中的碰撞网格地址改为相对地址。模型的质量、惯量、控制增益、物理步长、策略权重和保护条件没有因此改变。
- 历史说明和日志中可能出现旧目录、旧速度与旧部署状态，应按日期阅读。当前状态以根 README、活动 config 和策略 manifest 为准。

归档的 `verification.json`、`dependency_verification.json` 和元数据记录原私有整理包的来源；发布副本排除了私有交接说明、备份、缓存、压缩包及服务器连接脚本，因此不能把旧包的“所有文件完整”统计直接套用到本仓库。

## 训练与蒸馏

梯度训练在服务器 Isaac Sim 6.2 / Isaac Lab 3 / PhysX 中进行；这些环境不在推理 requirements 中。移动 PPO 的源码位于当前 step / wheel 策略目录；原始姿态和倒立训练代码位于 `server_snapshot/`。

针对性倒立蒸馏的入口为 `mujoco_latest/handstand_distill_20261112/targeted_driver.py`，数据采集脚本为 `collect_targeted.py`。目标引擎采样数据 `mujoco_targeted4179.npz`、1000 次更新快照和配对评估均已保留。训练器保留当时的 Linux 路径布局；在另一台训练机使用前，应准备对应 Isaac 环境，并按源码配置教师、数据和输出目录。它不是 `pip install -r requirements.txt` 后即可在纯 MuJoCo 环境运行的训练器。

本地 MuJoCo 提供教师标注样本与闭环验收，未在本地执行这次网络梯度训练。原训练完成后自动评估出现 inference tensor 重置错误，原退出码 1 保留；独立评估程序随后成功完成，不能把原任务改写为正常退出。

## 验证记录与发布检查

- `mujoco_latest/logs/handstand_targeted999_*.json`：六类场景、每类两次连续倒立与返回，无 reset。
- `mujoco_latest/handstand_distill_20261112/targeted_isaac20s.json`：256 并行、20 s 教师/学生配对评估。
- `mujoco_latest/handstand_distill_20261112/slip_comparison_targeted.json`：支撑偏移、接触滑移和动作误差诊断。
- `targeted_import_result.json`：实际导入与旧配置备份的来源记录。

发布副本另行检查七个策略入口均从仓库内部加载、模型总质量为 19.523 kg，以及抬腿/轮驱前进后停车和静止进入倒立—返回—再次倒立—返回—移动流程。发布检查产生的 `publish_*` 日志是本地生成文件，不加入版本控制。

当前结果属于有限场景的 Sim-to-Sim 验证。学生目标仍依赖足端/锚点信息，实机需要里程计或状态估计，并需验证传感器噪声、延迟、执行器模型与安全控制链路。
