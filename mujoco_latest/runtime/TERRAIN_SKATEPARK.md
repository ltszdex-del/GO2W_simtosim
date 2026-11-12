# 滑板公园测试地形

地图：`/home/ubuntu/Go2W/scenes/go2w_skatepark_20261107/go2w_skatepark.usd`。
测试环境：`/home/ubuntu/Go2W/interactive_stage1_20261104`。

`main.py` 使用 `terrain_setup.py` 按配置替换原 ground asset，挂载为 `/World/TestTerrain`。原无限平面不再生成，因此不会遮蔽坡面、台阶或导致同高度重复接触。地图的 19 个静态碰撞网格与物理材质沿用 USD 定义，碰撞组设为全局可接触；不额外创建 PhysicsScene。所有机器人、策略与控制参数保持原值。

默认在地图中央空地的 U 站姿出生。R 重置仍返回这个位置。地图中央 8×8 米为空地；整体范围 x∈[-18,18]、y∈[-14,14] 米，超过外边界会离开有限地面。

## 启动

在服务器执行（串流端口需空闲）：

```bash
cd /home/ubuntu/Go2W/interactive_stage1_20261104
bash start_wasd_stream.sh --terrain skatepark
```

WebRTC 地址仍为 `106.75.23.13:49100`。WASDQE、M 切换抬腿/轮驱、P/U、H 倒立/恢复以及 R 重置沿用现有接口。

切回原平地：

```bash
bash start_wasd_stream.sh --terrain flat
```

启动日志目录内的 `terrain.json` 记录本次实际选中的地形、源文件哈希、19 个碰撞路径、单位和出生点。

## 检查范围

`verify_terrain_contacts.py` 使用实际测试机器人和 U 站姿策略，检查中央地面接触、保持高度、原地重置，并通过 PhysX 射线查询检查地面、3cm 台阶及18cm平台的碰撞高度。验证结果记录在 `logs/skatepark_verify_*/validation.json`。

本次接入地形并不改变训练分布。现有移动策略主要在平地训练，不能据此保证上下台阶、弧坡、高落差的通过率。建议先测试中央空地、低台阶和缓坡，再根据失稳表现决定地形训练课程。

备份：`backups/skatepark_20261107/`（接入前的 main.py 与 config.json）。
