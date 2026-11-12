# P 从认可的 U1050 站姿初始化，从头训练200轮

U 项目：/home/ubuntu/Go2W/pu_ppo_u_vertical100_20261106
姿态来源：logs/U_vertical_resume100/policy_1049.pt，用户实际观看的1050轮策略，而非其后1100轮权重。

standing_initialization.json记录16–18秒100帧平均腿部与轮子角度、机身高度、横滚、俯仰；standing_source_samples.json保留原始采样。P reset采用该姿态，归一化XY/yaw、速度置零，高度加2mm数值间隙；不是加载U神经网络。P从随机网络开始训练200轮。

趴卧目标、自碰撞、三秒目标、全16关节泄力、水平±5°和冲击限制沿用原P快版。原U与原P项目均保留。新观测/动作279/12，日志logs/P_fromU1050_scratch200，每50轮保存，完成导出final.pt。
