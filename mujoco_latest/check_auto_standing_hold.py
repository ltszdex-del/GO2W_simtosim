"""Real MuJoCo stop/stand/restart and the measured-yaw/dwell gate."""
import bootstrap,json,torch
from run import create_runtime,advance
from low_speed_hold import LowSpeedHold

g=LowSpeedHold(bootstrap.load_config()['low_speed_hold'])
for _ in range(30):assert g.select('locomotion',[0,0,0],[0,0,1.],True) is None
assert g.settled_s==0.
for _ in range(30):assert g.select('locomotion',[.01,0,0],[0,0,0],True) is None
for _ in range(4):assert g.select('locomotion',[0,0,0],[0,0,0],True) is None
assert g.select('locomotion',[0,0,0],[0,0,0],True)=='standing_hold'
assert g.select('standing_hold',[.01,0,0],[0,0,0],True,motion_requested=True)=='locomotion'
for _ in range(30):assert g.select('locomotion',[0,0,0],[0,0,0],True,motion_requested=True) is None

rows=[]
for wheel in (False,True):
    for key in ('W','Q','E'):
        env,c=create_runtime()
        if wheel:c._set_mobility('wheel',env.snapshot())
        events=[];old=None
        with torch.inference_mode():
            for frame in range(650):
                keys=[key] if 100<=frame<175 or 400<=frame<440 else []
                before=env.snapshot()['actual_velocity_yaw_m_s_rad_s']
                c.feed('auto_hold_probe',keys);s=advance(env,c)
                if c.mode!=old:
                    if c.mode=='standing_hold' and frame>0:
                        assert max(abs(v) for v in c.command)<.001
                        assert (before[0]**2+before[1]**2)**.5<.15 and abs(before[2])<.15
                        assert c.low_speed_hold.settled_s+1.e-9>=.1
                    events.append(dict(frame=frame,mode=c.mode,actual=s['actual_velocity_yaw_m_s_rad_s']))
                    old=c.mode
                assert c.mode!='error',(wheel,key,frame,s['message'])
                if keys:assert c.mode=='locomotion',(wheel,key,frame,c.mode)
                if frame in (90,390,649):assert c.mode=='standing_hold',(wheel,key,frame,c.low_speed_hold.status(True))
                if c.mode=='standing_hold':assert env.motor_mode=='standing_hold'
        assert [e['mode'] for e in events].count('standing_hold')==3,events
        row=dict(wheel=wheel,key=key,passed=True,events=events);rows.append(row)
        print(json.dumps(row),flush=True)
(bootstrap.ROOT/'logs/auto_standing_hold_checks.json').write_text(json.dumps(rows,indent=2))
