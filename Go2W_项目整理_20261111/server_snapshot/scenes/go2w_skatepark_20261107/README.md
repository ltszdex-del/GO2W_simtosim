# Go2W 滑板公园式训练场

独立静态场景，36×28米，Z向上、米制，中央8×8米无遮挡。共14组障碍，19个静态碰撞网格。没有机器人，也不绑定任何训练任务。

## 打开

- `go2w_skatepark.usd`：用于引用到后续机器人训练场景；不自带PhysicsScene，避免与训练环境的物理设置冲突。
- `preview.usda`：在Isaac Sim中直接打开查看，包含相对引用、灯光、相机和重力场景。可切换到`/World/Camera`查看全景。
- `go2w_skatepark.usda`：同一地形的可读文本版本。
- `terrain_manifest.json`：尺寸、各台阶累计高度、踏面深度、障碍入口位置、投放区域和碰撞参数。

远程目录：`/home/ubuntu/Go2W/scenes/go2w_skatepark_20261107`。

## 障碍

| 编号 | 类型 | 尺寸/难度 |
|---|---|---|
| S01 | 微型连续台阶 | 单级3cm，6级，总高18cm |
| S02 | 低台阶 | 单级6cm，5级，总高30cm |
| S03 | 标准台阶 | 单级10cm，5级，总高50cm |
| S04 | 高台阶 | 单级15cm，4级，总高60cm |
| S05 | 不等高、不等深台阶 | 单级4/8/6/12/8cm，踏深28–90cm |
| S06 | 宽踏面大台阶 | 单级18cm，3级，踏深90/90/120cm |
| S07 | 上下双向阶梯 | 8cm一级，中央32cm平台，再逐级下降 |
| R01 | 缓坡双向平台 | 高25cm，坡度约5.7° |
| R02 | 桌台坡 | 高50cm，两侧直坡 |
| R03 | 陡坡平台 | 高60cm，坡度约21.8° |
| R04 | 四面坡台 | 底5×5m，顶1.6×1.6m，高45cm |
| L01 | 低矮长平台 | 高12cm，3×1.5m |
| Q01 | 四分之一弧坡 | 半径1m，高1m，96段弧面 |
| Q02 | 小型半管 | 高80cm，半径80cm，中间3m平地 |

S01–S06及L01的后端为垂直落差；Q01/Q02有高处边缘。它们是不同难度的静态障碍，不是安全通过保证。训练可先开放S01/S02/R01/L01，再逐步加入其他区域。入口/出口示例仅给出几何参考，根高度0.42m需随机器人站姿调整；弧坡和高台的入口并非全部可直接平地步行到达。

## 训练引用

可以使用当前Isaac Lab项目已有的`UsdFileCfg`，例如在环境的`/World/Terrain`挂载本文件：

```python
terrain = AssetBaseCfg(
    prim_path="/World/Terrain",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/Go2W/scenes/go2w_skatepark_20261107/go2w_skatepark.usd"
    ),
)
```

不要再放置同高度的重叠地板。若要并行复制完整地形，实例间距建议不小于40m；也可以把多个机器人放在一份共享地形上，并由任务代码处理环境间碰撞隔离。重置时只在清单的平地区域采样，地形任务需另外提供高度/接触观测、目标方向、终止和奖励；本次没有改动原有U/P或跌倒恢复策略。

## 碰撞与验证

障碍均为封闭、方向一致的静态三角网格，使用`UsdPhysics.CollisionAPI`及`MeshCollisionAPI(approximation=none)`，适用于静态凹形半管和台阶，不能直接转换为单个凸包，否则地形会失真。装饰标线无碰撞。静摩擦0.9，动摩擦0.8，恢复系数0。

`validation.json`记录USD结构检查：网格闭合与绕向、无退化面、材质绑定、中央区域净空、单位和自包含依赖。机器人实际接触、可通行性与训练效果需要在后续任务中验证。

`build_scene.py`可重建场景（需要pxr与numpy）；`make_map.py`生成布局图；`render_preview.py`使用当前Isaac Sim渲染实际USD。

当前服务器渲染命令：

```bash
cd /home/ubuntu/Go2W/scenes/go2w_skatepark_20261107
/home/ubuntu/envs/isaac62/bin/python render_preview.py --livestream 0 --visualizer kit --kit_args=--no-window
```
