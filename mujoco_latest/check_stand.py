"""Measure stance vibration and verify the held pose releases to movement."""
import bootstrap,json,torch,numpy as np
from run import create_runtime,advance
env,c=create_runtime();rows=[]
with torch.inference_mode():
    for k in range(700):
        if k==200:c._enter_standing_hold(env.snapshot())
        c.feed('stand_probe',[]);s=advance(env,c)
        rows.append([k,float(env.robot.data.joint_vel.torch[0,:12].square().mean().sqrt()),float(env.torque.abs().max()),float(env.robot.data.root_ang_vel_b.torch.norm()),float(env.target[0,0])])
    assert c.mode=='standing_hold'
    for k in range(60):
        c.feed('stand_probe',['W']);s=advance(env,c)
    assert c.mode=='locomotion' and s['physics']['xyz'][2]>.18
metrics={}
for name,lo,hi in [('step_hold',100,200),('transition',200,225),('standing_hold',500,700)]:
    x=np.array(rows[lo:hi]);metrics[name]=dict(dq_rms=float(x[:,1].mean()),dq_peak=float(x[:,1].max()),torque_peak=float(x[:,2].max()),omega_rms=float(np.sqrt(np.mean(x[:,3]**2))))
    print(name,metrics[name],flush=True)
(bootstrap.ROOT/'logs/stand_after.json').write_text(json.dumps(dict(metrics=metrics,samples=rows),indent=2))
