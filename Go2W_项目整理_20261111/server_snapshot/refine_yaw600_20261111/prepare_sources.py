from pathlib import Path
import shutil

h=Path(__file__).resolve().parent
for kind,stationary,cruising,accel,brake in [('wheel',4.,2.,4.8,19.2),('step',3.14,1.6,4.415625,7.065)]:
    root=h/kind
    shutil.copy2(h/'yaw_envelope.py',root/'yaw_envelope.py')
    p=root/'config.py'
    p.write_text(p.read_text()+f'\n# Final capability refinement: reference translation determines yaw ceiling.\nCOMMAND.update(yaw_speed_limit={stationary!r},yaw_cruise_limit={cruising!r},yaw_speed_decay=3.,yaw_acceleration={accel!r},yaw_brake={brake!r})\n')
    p=root/'schedule.py';s=p.read_text()
    s=s.replace('from directional_reference import planar_tensor','from directional_reference import planar_tensor\nfrom yaw_envelope import yaw_limit_tensor,brake_to_limit_tensor')
    start=s.index('        self.velocity[:,2:3]=planar_tensor')
    end=s.index('        self.heading=wrap',start)
    s=s[:start]+('''        cap=getattr(self,'speed_cap',COMMAND['speed_limits'][0])
        self.velocity[:,0].clamp_(min=-cap,max=cap)
''' if kind=='wheel' else '')+'''        limit=yaw_limit_tensor(self.velocity,COMMAND['speed_limits'][0],COMMAND['min_forward_speed'],
            COMMAND['speed_limits'][1],COMMAND['yaw_speed_limit'],COMMAND['yaw_cruise_limit'],COMMAND['yaw_speed_decay'])
        self.current_yaw_limit=torch.minimum(limit,torch.as_tensor(getattr(self,'yaw_cap',COMMAND['yaw_speed_limit']),device=limit.device))
        candidate=planar_tensor(old[:,2:3],axes[:,2:3],dt,
            [COMMAND['yaw_acceleration']],[COMMAND['yaw_brake']],[1.e9])[:,0]
        self.velocity[:,2]=brake_to_limit_tensor(candidate,old[:,2],self.current_yaw_limit,COMMAND['yaw_brake'],dt)
'''+s[end:]
    p.write_text(s)
    p=root/'learning.py';s=p.read_text()
    if kind=='wheel':
        s=s.replace('env.turn_cap=2.','env.turn_cap=4.').replace('yaw_course_cap=2.,','yaw_course_cap=4.,')
    else:
        s=s.replace('env.qe_speed_cap=.8','env.qe_speed_cap=3.14').replace('env.qe_speed_cap=1.6','env.qe_speed_cap=3.14')
    s=s.replace('row=dict(iteration=iteration,','row=dict(yaw_limit_mean=float(env.command.current_yaw_limit.mean()),yaw_limit_min=float(env.command.current_yaw_limit.min()),yaw_limit_max=float(env.command.current_yaw_limit.max()),iteration=iteration,')
    p.write_text(s)
    if kind=='step':
        p=root/'env.py';p.write_text(p.read_text().replace("getattr(self,'qe_speed_cap',1.6) if self.cfg.training else 1.6","getattr(self,'qe_speed_cap',3.14) if self.cfg.training else 3.14"))
shutil.copy2(h/'yaw_envelope.py',h/'runtime/yaw_envelope.py')
