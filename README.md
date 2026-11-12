# GO2W Sim-to-Sim

基于强化学习的 Go2-W 轮足机器人多技能控制与 **Isaac Sim → MuJoCo** 验证项目。集成抬腿移动、轮驱移动、站立保持、卧倒、起立、倒立及倒立返回，包含训练源码快照、当前策略权重、状态机、模型转换工具和验证记录。

Personal robotics project for PPO-based wheeled-quadruped control and cross-simulator validation, including targeted DAgger distillation for handstands. Interactive use has been tested on Windows; this repository does not claim real-robot deployment or general terrain robustness.

## 快速运行

推荐使用独立 Python 3.9 环境；已验证 MuJoCo 3.3.2、PyTorch 2.8.0 CPU。安装 PyTorch 时也可按所需平台使用其官方安装方式。

```powershell
git clone https://github.com/ltszdex-del/GO2W_simtosim.git
cd GO2W_simtosim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python mujoco_latest/run.py
```

提交中已包含场景模型、渲染网格及当前权重。常规运行无需安装 Isaac Sim、连接服务器或重新生成模型。程序从自身目录解析路径；不要单独移动 `mujoco_latest`，其他目录也是运行依赖。

| 按键 | 功能 |
|---|---|
| W / S / A / D | 前后与横向移动 |
| Q / E | 转向 |
| M | 抬腿 / 轮驱切换 |
| H | 倒立 / 倒立返回 |
| P / U | 卧倒 / 起立 |
| R | Reset |
| K | 当前版本固定使用 LOW 指令范围 |

Windows 下让仿真窗口获得焦点后操作。绿色箭头表示参考速度，蓝色表示实际速度，竖直箭头表示角速度。本地网页控制面板为 `http://127.0.0.1:8776`。其他系统的键盘交互未经验证，可使用面板或无界面入口。

无界面检查：

```powershell
python mujoco_latest/run.py --headless --steps 350 --scenario W --log publish_smoke
python mujoco_latest/check_handstand_fix.py --case idle --result-tag publish_handstand
```

## 当前策略与结果

版本整理日期：2026-11-12。七个入口对应六份独立策略权重，起立与站立保持复用权重；倒立适配与返回还保留原教师依赖。

| 控制项 | 当前版本 |
|---|---|
| 抬腿移动 | `policy_10474.pt` |
| 轮驱移动 | `policy_3824.pt`，已回退至提速实验之前 |
| 倒立 | `handstand_targeted_20261112/.../student_999.pt`，原教师 4179 的针对性蒸馏学生 |
| 倒立返回 | `policy_0349.pt` |
| 卧倒、起立 / 站立保持 | 原 P / U `final.pt` |

主要工作：

- PPO 训练、镜像对称约束、轮腿联合动作、组合指令与制动课程。
- 观测、关节顺序、坐标系、执行器、历史状态与策略切换的跨引擎适配；使用源码及资产哈希校验接口。
- 通过分步长对照定位振荡，对站立及姿态技能使用 1 ms 电机/物理内环，保留 50 Hz 策略更新。
- 针对倒立学生在 MuJoCo 约 4.88 s 失稳的问题，采集目标引擎实际访问状态的教师标签，结合分层 DAgger 和关键动作加权补训；通过六类场景、共 12 次连续倒立与返回测试。

对应诊断阶段的前轮支撑锚点偏移峰值由旧学生约 0.391 m 降至 0.168 m，接近教师 0.169 m。该指标含正常滚动，不等于纯接触打滑距离；当前回归结果不代表任意扰动与实机成功率。

## 文档与代码结构

- [完整 Sim-to-Sim 项目说明](mujoco_latest/申校材料/Go2W_Sim2Sim_项目说明.md)
- [倒立针对性蒸馏问题与经验复盘](mujoco_latest/申校材料/倒立针对性蒸馏复盘_20261112.md)
- [中英文项目条目](mujoco_latest/申校材料/Go2W_简历项目条目_中英.md)
- [训练方法与源码阅读指南](Go2W_项目整理_20261111/docs/02_目录结构与代码阅读指南.md)
- [当前权重及来源](docs/REPRODUCIBILITY.md)
- [第三方来源与声明](THIRD_PARTY_NOTICES.md)

```text
mujoco_latest/                   # MuJoCo 执行、适配器、状态机、验证
  runtime/                      # 当前交互控制代码
  policies/                     # 当前权重、相关源码与部署记录
  generated/                    # 已生成场景与原始碰撞网格
  handstand_distill_20261112/    # 蒸馏、采样、诊断与验收记录
  申校材料/                     # 项目说明、复盘与中英文摘要
Go2W_项目整理_20261111/           # 日期明确的训练源码/依赖快照及技术文档
audit_sim2sim_20261104/staged/    # 保留惯量的 URDF 转换器
.tmp_unitree_mujoco/             # 程序引用的 Unitree 渲染网格依赖
third_party/                    # 上游许可文本
```

训练在服务器 Isaac Sim / Isaac Lab / PhysX 中完成，本地 MuJoCo 用于数据采集与验证。`requirements.txt` 仅覆盖本地推理；训练环境和重建模型见复现说明。原始训练快照按日期保存，当前运行版本以本页与 `mujoco_latest/runtime/config.json` 为准。
