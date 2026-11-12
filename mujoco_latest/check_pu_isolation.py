"""Exercise repeated P/U after motion, with real state transitions and no reset."""
import bootstrap
import json,sys
import torch
from run import create_runtime, advance

results=[]
for wheel in ((True,) if '--wheel' in sys.argv else (False,True)):
    env,c=create_runtime()
    if wheel:c._set_mobility('wheel',env.snapshot())
    events=[];previous=None;frame=0
    def tick(keys=()):
        global frame,previous
        c.feed('pu_isolation_probe',list(keys))
        status=advance(env,c);frame+=1
        if c.mode!=previous:
            events.append(dict(frame=frame,mode=c.mode,motor=env.motor_mode))
            previous=c.mode
        if c.mode=='error':
            ctx=env.handstand_context
            print('FAIL_STATE',json.dumps({k:v.tolist() for k,v in ctx.metrics().items() if k in ('q','roll','pitch','v','w','z','qe')}),flush=True)
        assert c.mode!='error',(wheel,frame,status['message'])
        assert env.model.opt.timestep==env.physics_dt
        expected=20 if getattr(env.handstand_context,'local_pu_isolation',False) else 4
        assert env.pd_updates==expected,(c.mode,env.pd_updates,expected)
        return status
    def wait_mode(mode,budget):
        for _ in range(budget):
            tick()
            if c.mode==mode:return
        ctx=env.handstand_context
        print('TIMEOUT_STATE',{k:v.tolist() for k,v in ctx.metrics().items() if k in ('roll','pitch','qe','v','w','z','leg_speed','belly','body_mean_bw')},'gain',ctx.gain.tolist(),'ready',c.active.ready_time,flush=True)
        raise AssertionError((wheel,'timeout',mode,c.mode,c.message,events))
    with torch.inference_mode():
        for _ in range(100):tick()
        for cycle,key in enumerate(('W','Q','S')):
            for _ in range(50):tick([key])
            for _ in range(80):tick()
            tick(['P']);wait_mode('prone',500)
            speeds=[]
            for _ in range(50):
                tick();speeds.append(float(env.robot.data.joint_vel.torch[0,:12].square().mean().sqrt()))
            tick(['U']);wait_mode('standing_hold',500)
            standing=[]
            for _ in range(100):
                tick();standing.append(float(env.robot.data.joint_vel.torch[0,:12].square().mean().sqrt()))
            results.append(dict(wheel=wheel,cycle=cycle,lying_dq_rms=sum(speeds)/len(speeds),
                                standing_dq_rms=sum(standing[-50:])/50,events=events.copy()))
            print(json.dumps(results[-1]),flush=True)
        for _ in range(40):tick(['W'])
    assert c.mode=='locomotion'
(bootstrap.ROOT/'logs/pu_isolation_checks.json').write_text(json.dumps(results,indent=2))
