import argparse,json,ast,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--num_envs',type=int,default=16)
parser.add_argument('--steps',type=int,default=650)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.headless=True
app=AppLauncher(args).app
env=None
try:
    import torch
    from settings import ROOT,ACTOR_DIM
    from env import ReturnCfg,ReturnEnv
    from ppo_math import masked_gae
    torch.set_num_threads(4)
    # A large reward during teacher ownership must never enter actor GAE.
    r=torch.tensor([[1000.],[2.],[3.]])
    done=torch.tensor([[False],[False],[True]])
    controlled=torch.tensor([[False],[True],[True]])
    adv,_=masked_gae(r,done,torch.zeros_like(r),torch.zeros(1),controlled)
    assert adv[0]==0 and torch.allclose(adv[1],torch.tensor([2+.995*.95*3])) and adv[2]==3
    tree=ast.parse((ROOT/'env.py').read_text())
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and cls.name=='ReturnEnv':
            for method in cls.body:
                if isinstance(method,ast.FunctionDef):
                    for node in ast.walk(method):
                        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr.startswith('write_') and '_to_sim' in node.func.attr:
                            assert method.name=='_reset_idx',(method.name,node.func.attr)
    cfg=ReturnCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=42;cfg.sim.device=args.device or 'cuda:0'
    env=ReturnEnv(cfg);obs,_=env.reset()
    assert obs['policy'].shape==(args.num_envs,ACTOR_DIM)
    assert torch.all(env.robot.data.joint_damping.torch[:,env.wheels]==0)
    audited_handoffs=[0]
    original=env._handoff
    def audit(mask,m):
        if mask.any():
            d=env.robot.data
            before=(d.root_state_w.torch.clone(),d.joint_pos.torch.clone(),d.joint_vel.torch.clone())
            original(mask,m)
            after=(d.root_state_w.torch,d.joint_pos.torch,d.joint_vel.torch)
            assert all(torch.equal(a,b) for a,b in zip(before,after)),'Handoff changed physical state'
            audited_handoffs[0]+=int(mask.sum())
        else:original(mask,m)
    env._handoff=audit
    rows=[];controlled_samples=0;episodes=[];min_pitch=2.;max_rear=0.;max_torque=0.
    with torch.inference_mode():
        for i in range(args.steps):
            obs,reward,term,trunc,info=env.step(torch.zeros(args.num_envs,16,device=env.device))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all() and torch.isfinite(env.torque).all()
            assert env.pd_updates==4 and env.h._pd_substeps==4 and env.u.pd_count==4
            warm=(~env.last_step_mask)&~(term|trunc)
            if warm.any():assert torch.equal(env.torque[warm],env.h.torque[warm]),'Warm-up motor law changed'
            controlled_samples+=int(env.last_step_mask.sum())
            if 'episode' in info:episodes.append(dict(info['episode']))
            if env.last_step_mask.any():
                min_pitch=min(min_pitch,float(env.m['pitch'][env.last_step_mask].amin()))
                max_rear=max(max_rear,float((env.landing_time>=0).all(-1).float().mean()))
            max_torque=max(max_torque,float(env.torque.abs().max()))
            if i%100==0:rows.append(dict(step=i,**info.get('log',{})))
    assert audited_handoffs[0]>=args.num_envs and controlled_samples>0,(audited_handoffs,controlled_samples)
    result=dict(passed=True,controlled_samples=controlled_samples,handoffs=audited_handoffs[0],
        teacher_resets=env.teacher_resets,max_torque_nm=max_torque,minimum_return_pitch_deg=min_pitch*180/3.14159265,
        max_rear_landed_fraction=max_rear,episodes=episodes,trace=rows,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('env.py','settings.py','teacher_context.py','model.py','ppo_math.py','train.py')},
        checks=['no in-episode state writes','handoff preserves full physical state','teacher torque exact before handoff',
            'warm-up excluded from GAE','four PD substeps','finite observations, rewards and torques','zero wheel implicit damping'])
    (ROOT/'smoke_result.json').write_text(json.dumps(result,indent=2))
    print('SMOKE',json.dumps({k:v for k,v in result.items() if k not in ('episodes','trace')}),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    if env:env.close()
    app.close(exit_code=1)
else:
    if env:env.close()
    app.close()
