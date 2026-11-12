"""Production H: long hold, return, repeat, then resume mobility without reset."""
import bootstrap,argparse,json
import numpy as np
import torch
from run import create_runtime,advance

p=argparse.ArgumentParser();p.add_argument('--case',choices=['idle','W','Q','E','wheel'],default='idle')
p.add_argument('--approach-frames',type=int,default=50)
p.add_argument('--result-tag',default='handstand_fix')
args=p.parse_args();env,c=create_runtime();events=[];old=None;frame=0;stands=[];held=[]
if args.case=='wheel':c._set_mobility('wheel',env.snapshot())
def tick(keys=()):
    global frame,old
    c.feed('H_fix_check',list(keys));s=advance(env,c);frame+=1
    if c.mode!=old:
        events.append(dict(frame=frame,mode=c.mode,motor=env.motor_mode));old=c.mode
    assert c.mode!='error',(args.case,frame,c.message)
    # The controller can hand off after physics, so check update counts only
    # when the mode did not change in after_step.
    if env.pd_updates not in (4,20):raise AssertionError(env.pd_updates)
    assert env.model.opt.timestep==.005
    return s
def wait_mode(mode,budget=500):
    for _ in range(budget):
        tick()
        if c.mode==mode:return
    raise AssertionError((args.case,'timeout',mode,c.mode,c.message,events))
with torch.inference_mode():
    for _ in range(100):tick()
    if args.case in ('W','Q','E','wheel'):
        for _ in range(args.approach_frames):tick(['W' if args.case=='wheel' else args.case])
        for _ in range(100):tick()
    for cycle,duration in enumerate((1000,500)):
        tick(['H']);wait_mode('handstand')
        speeds=[];entry_heading=float(c.adapters['handstand'].ctx.stable_heading()[0])
        for k in range(duration):
            tick();speeds.append(float(env.robot.data.joint_vel.torch[0,6:12].abs().max()))
            assert c.mode=='handstand' and env.pd_updates==20
            if k>200:assert env.robot.data.projected_gravity_b.torch[0,0]>.9
        held.append(dict(cycle=cycle,seconds=duration*.02,entry_heading_rad=entry_heading,peak_rear_speed=max(speeds)))
        tick(['H']);wait_mode('standing_hold')
        dq=[]
        for _ in range(100):
            tick();dq.append(float(env.robot.data.joint_vel.torch[0,:12].square().mean().sqrt()))
        stands.append(float(np.mean(dq[-50:])))
        assert stands[-1]<.1,(args.case,'standing vibration',stands[-1])
    for _ in range(50):tick(['W'])
    assert c.mode=='locomotion' and env.pd_updates==4
    for _ in range(250):tick()
    assert c.mode=='standing_hold'
result=dict(case=args.case,approach_frames=args.approach_frames,passed=True,held=held,standing_leg_rms=stands,events=events)
suffix='' if args.approach_frames==50 else '_turn'+str(args.approach_frames)
(bootstrap.ROOT/'logs'/(args.result_tag+'_'+args.case+suffix+'.json')).write_text(json.dumps(result,indent=2))
print('PASS',json.dumps(result),flush=True)
