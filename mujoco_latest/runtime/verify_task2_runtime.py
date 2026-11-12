"""Finite headless integration check; never installs a test adapter in main.py."""
import argparse,json,time,urllib.request,math
from pathlib import Path
from isaaclab.app import AppLauncher
from control import Controller
from bridge import serve

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--log_dir',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
folder=Path(args.log_dir);folder.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'config.json').read_text());cfg['bridge_port']=18765
launcher=AppLauncher(args);app=launcher.app;env=server=None
try:
    import torch
    from scene import StandingEnv,EnvCfg
    ec=EnvCfg();ec.sim.device=args.device or 'cuda:0'
    env=StandingEnv(ec,cfg);env.reset()
    clock=[0.];c=Controller(cfg,lambda:clock[0]);c.contract=env.contract()
    server=serve(c)
    def post(keys=None,release=False):
        payload={'session':'task2-runtime','keys':keys or []}
        request=urllib.request.Request('http://127.0.0.1:18765/'+('release' if release else 'input'),
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=2) as response:assert response.status==204
    zeros=torch.zeros((1,16),device=env.device)
    rows=[];cases={};start=time.monotonic()
    with torch.inference_mode():
        for name,keys in [('W',['W']),('S',['S']),('A',['A']),('D',['D']),('Q',['Q']),('E',['E']),('WAQ',list('WAQ')),('WS',list('WS'))]:
            post(release=True)
            for _ in range(80):
                clock[0]+=.02;post(keys)
                state=env.snapshot();c.tick(.02,state);env.step(zeros)
                assert env.pd_updates==4
                rows.append({'case':name,'reference':c.command[:],'actual':env.snapshot()['actual_velocity_yaw_m_s_rad_s'],
                    'xyz':env.robot.data.root_pos_w.torch[0].tolist(),'peak_torque_nm':float(env.torque.abs().max())})
            cases[name]=c.status()
        post([])
        for _ in range(50):clock[0]+=.02;c.tick(.02,env.snapshot());env.step(zeros)
        assert c.command==[0.,0.,0.]
        post(['W']);c.tick(.02,env.snapshot());clock[0]+=.41;c.tick(.02,env.snapshot())
        assert c.command==[0.,0.,0.] and c.stats['timeouts']==1
        post(release=True)
        class ProbeAdapter:
            path='test-only://physical-target-pipeline'
            manifest={**c.contract,**{k:None for k in ['observation_order','normalization','scales','history','phase','previous_action','target','task_id']}}
            def __init__(self):self.count=0;self.calls={k:0 for k in ['reset','observe','infer','apply','is_complete']}
            def load(self):pass
            def reset(self,state):self.calls['reset']+=1
            def observe(self,state):self.calls['observe']+=1;return state
            def infer(self,state):self.calls['infer']+=1;return self.count
            def apply(self,action,dt):
                self.calls['apply']+=1;self.count+=1
                q=[0.,.8,-1.5]*4;q[1]+=.015*math.sin(self.count*.05)
                return {'leg_pos_rad':q,'wheel_vel_rad_s':[0.]*4}
            def is_complete(self,state):self.calls['is_complete']+=1;return self.count>=120
        probe=ProbeAdapter();c.register('lying_down',probe)
        # Test the real exclusive execution path, using a tiny non-task target
        # perturbation, never labeling this probe as a successful P policy.
        c._activate('lying_down',env.snapshot())
        maximum_target_delta=0.
        for _ in range(120):
            clock[0]+=.02;state=env.snapshot();c.tick(.02,state);state['command']=c.status()
            command=c.execute(state,.02);env.set_motor_targets(command);env.step(zeros)
            maximum_target_delta=max(maximum_target_delta,float(abs(env.target[0,1]-.8)))
            c.after_step(env.snapshot());assert env.pd_updates==4
        assert c.mode=='prone' and c.active is None and maximum_target_delta>.005
        env.standing_target()
        for _ in range(120):env.step(zeros)
        final=env.snapshot()
        assert float(env.robot.data.root_pos_w.torch[0,2])>.35
        assert torch.isfinite(env.robot.data.joint_pos.torch).all()
    report=dict(passed=True,cases=cases,manifest=c.contract,probe_calls=probe.calls,
        probe_max_target_delta_rad=maximum_target_delta,probe_is_not_a_task_policy=True,
        final_height_m=float(env.robot.data.root_pos_w.torch[0,2]),wall_s=time.monotonic()-start,
        standing_only_real_mobility_acceptance=False,
        final_actual_velocity=final['actual_velocity_yaw_m_s_rad_s'])
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (folder/'trace.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    print('[TASK2_RUNTIME_PASS]',json.dumps(report,ensure_ascii=False),flush=True)
finally:
    if server:server.shutdown();server.server_close()
    if env:env.close()
    app.close()
