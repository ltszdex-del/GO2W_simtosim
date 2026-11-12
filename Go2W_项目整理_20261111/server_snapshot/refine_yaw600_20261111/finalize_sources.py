from pathlib import Path
h=Path(__file__).resolve().parent
for name in ('run_one.py','launch.py','check_schedule.py','training_snapshot.py'):
    p=h/name;s=p.read_text().replace('refine_combo300_20261111','refine_yaw600_20261111').replace('combo300_20261111','yaw600_20261111')
    if name=='run_one.py':
        s=s.replace('updates=300','updates=600').replace('goal=2699 if kind==\'wheel\' else 9899',"goal=3299 if kind=='wheel' else 10499")
    if name=='launch.py':s=s.replace("h/'run_one.py'","h/'queue_one.py'")
    if name=='check_schedule.py':s=s.replace("root=Path('/home/ubuntu/Go2W')/f'go2w_{kind}_yaw600_20261111'","root=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/home/ubuntu/Go2W')/f'go2w_{kind}_yaw600_20261111'")
    if name=='training_snapshot.py':s=s.replace("[('wheel',2399),('step',9599)]","[('wheel',2699),('step',9899)]")
    p.write_text(s)
p=h/'runtime/controls_panel.html';s=p.read_text(encoding='utf-8')
s=s.replace('当前无运动策略，命令尚不驱动移动。','M 切换抬腿/轮驱，K 切换高/低速（上限减半、加速度不变）。')
s=s.replace('<button id="stop">','<button data-key="M">M 抬腿/轮驱</button><button data-key="K">K 高/低速</button><button id="stop">')
s=s.replace("'WASDQEHPU'.includes(k)","'WASDQEHPUMK'.includes(k)")
s=s.replace('${s.mobility_profile} ｜ 前进上限','${s.mobility_profile} / ${s.speed_mode} ｜ 前进上限')
s=s.replace('±${s.command_limits.max_yaw_rad_s} rad/s','±${s.command_limits.current_yaw_rad_s.toFixed(2)} rad/s（零平移 ${s.command_limits.max_yaw_rad_s}）')
p.write_text(s,encoding='utf-8')
