"""Check new deployment against training observation/target code in the live test scene."""
import argparse,ast,json,traceback,sys,math
from pathlib import Path
from types import SimpleNamespace as NS
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--log_dir',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
r=Path(__file__).resolve().parent;folder=Path(a.log_dir);folder.mkdir(parents=True,exist_ok=True)
cfg=json.loads((r/'config.json').read_text());cfg['policies']['wheel_locomotion']['factory']='wheel_rear_idle_candidate:create';cfg['wheel_idle_assist']=False;launcher=AppLauncher(a);app=launcher.app;env=None
try:
    import torch
    torch.set_num_threads(4)
    from scene import StandingEnv,EnvCfg,LEGS,WHEELS
    from control import Controller
    from plugin_adapters import load_adapter
    from terrain_setup import apply_terrain,inspect_loaded_terrain
    ec=EnvCfg();ec.sim.device=a.device or 'cuda:0';ec.scene.robot.spawn.asset_path=str(Path(cfg['robot_asset_root'])/'urdf/go2w_description.urdf')
    initial=cfg['initial_standing'];ec.scene.robot.init_state.pos=tuple(initial['root_pose'][:3]);ec.scene.robot.init_state.rot=tuple(initial['root_pose'][3:])
    ec.scene.robot.init_state.joint_pos=dict(zip(LEGS+WHEELS,initial['q']+initial['wheel_q']))
    ec.scene.robot.actuators['motors'].damping={'.*_hip_joint':0.,'.*_thigh_joint':0.,'.*_calf_joint':0.,'.*_foot_joint':100.}
    terrain=apply_terrain(ec,cfg);env=StandingEnv(ec,cfg);env.reset();terrain=inspect_loaded_terrain(terrain)
    now=[0.];c=Controller(cfg,lambda:now[0]);c.contract=env.contract()
    for mode in ('wheel_locomotion','handstand','handstand_return','lying_down','getting_up','standing_hold'):c.policy_contracts[mode]=env.contract(mode)
    for mode,spec in cfg['policies'].items():
        ad=load_adapter(spec,r,env.device);c.register(mode,ad)
        if hasattr(ad,'bind_env'):ad.bind_env(env)
    def source_functions(mode):
        spec=cfg['policies'][mode];ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        tree=ast.parse((Path(spec['training_root'])/'env.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionEnv')
        frame=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='frame')
        body=[frame]
        if mode=='wheel_locomotion':
            pre=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
            start=next(i for i,n in enumerate(pre.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='goal')
            fn=ast.FunctionDef(name='target_step',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=pre.body[start:],decorator_list=[])
            body.append(fn)
        glob={'torch':torch,'GAIT':ck['manifest']['phase'],'C':ck['manifest']['control'],'reference_phase':lambda age,period:age*(2*torch.pi/period)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'training_reference','exec'),glob)
        return glob
    sources={m:source_functions(m) for m in ('locomotion','wheel_locomotion')}
    errors={};action=torch.zeros(1,16,device=env.device)
    with torch.inference_mode():
        for profile in ('stepping','wheel','stepping'):
            env.handstand_context=None;env.reset();env.target=torch.tensor(cfg['initial_leg_target'],device=env.device)[None];env.goal=env.target.clone();env.wheel_target.zero_()
            c.reset_runtime(env.snapshot());c._set_mobility(profile,env.snapshot())
            maxframe=maxwheel=maxleg=0.;samples=0
            for k in range(140):
                keys=['W'] if 10<=k<25 else ['Q'] if 60<=k<75 else []
                now[0]+=.02;c.feed('import-test',keys);state=env.snapshot();c.tick(.02,state);state['command']=c.status()
                active=c.active
                if c.mode=='locomotion':
                    mode='wheel_locomotion' if profile=='wheel' else 'locomotion';functions=sources[mode]
                    obs=active.observe(state);angle=c.integrator.yaw_target-state['yaw_rad']
                    ctx=NS(robot=env.robot,leg_ids=env.legs,ids=env.ids,stand=active.stand,
                        action=active.previous,command=NS(velocity=torch.tensor([c.command],device=env.device)),
                        gait_period=torch.tensor([active.ck['manifest']['phase']['period']],device=env.device),phase_offset=torch.zeros(1,device=env.device),episode_length_buf=torch.tensor([round(active.age/.02)],device=env.device),step_dt=.02)
                    expected=functions['frame'](ctx,{'heading_error':torch.tensor([math.atan2(math.sin(angle),math.cos(angle))],device=env.device)})
                    delta=float((obs.reshape(1,10,57)[:,-1]-expected).abs().max());maxframe=max(maxframe,delta);assert delta<5e-5,(profile,delta)
                    net=active.infer(obs)
                    if profile=='wheel':
                        y=active.wheel_center_y();feet=torch.zeros(1,4,3,device=env.device);feet[:,:,1]=y
                        reference=NS(stand=active.stand,scale=active.scale,action=net,limits=active.limits,
                            target=active.target.clone(),wheel_target=active.wheel_target.clone(),step_dt=.02,
                            command=ctx.command,physical=lambda:dict(feet_b=feet))
                        functions['target_step'](reference)
                    motor=active.apply(net,.02);samples+=1
                    if profile=='wheel':
                        dw=float((active.wheel_target-reference.wheel_target).abs().max());dl=float((active.target-reference.target).abs().max())
                        maxwheel=max(maxwheel,dw);maxleg=max(maxleg,dl);assert dw<1e-5 and dl<1e-6,(dw,dl)
                else:motor=c.execute(state,.02)
                if motor is not None:env.set_motor_targets(motor)
                env.step(action);c.after_step(env.snapshot())
                assert env.pd_updates==4 and torch.isfinite(env.robot.data.joint_pos.torch).all()
            assert samples>10,(profile,samples)
            errors[profile]=dict(samples=samples,max_frame_error=maxframe,max_wheel_target_error=maxwheel,max_leg_target_error=maxleg)
    recovery_results=[]
    cfg['wheel_idle_assist']=True
    with torch.inference_mode():
        for key in ('W','S','Q','E'):
            env.handstand_context=None;env.reset()
            env.target=torch.tensor(cfg['initial_leg_target'],device=env.device)[None]
            env.goal=env.target.clone();env.wheel_target.zero_()
            c.reset_runtime(env.snapshot());c._set_mobility('wheel',env.snapshot())
            minimum_blend=1.;max_rate=0.;previous=None
            for k in range(400):
                now[0]+=.02;c.feed('recovery-check',[key] if k<40 else [])
                state=env.snapshot();c.tick(.02,state);state['command']=c.status()
                motor=c.execute(state,.02)
                if motor is not None:env.set_motor_targets(motor)
                if c.mode=='locomotion':
                    if previous is not None:max_rate=max(max_rate,float((c.active.target-previous).abs().max())/.02)
                    previous=c.active.target.clone()
                env.step(action);c.after_step(env.snapshot())
                assert c.mode!='error' and env.robot.data.root_pos_w.torch[0,2]>.19,(key,k,float(env.robot.data.root_pos_w.torch[0,2]))
            active=c.active;d=env.robot.data
            neutral=torch.tensor(cfg['initial_leg_target'],device=env.device)[None]
            reference=torch.tensor(cfg['initial_standing']['q'],device=env.device)[None]
            pose_rms=float((d.joint_pos.torch[:,env.legs]-reference).square().mean().sqrt())
            target_error=float((active.target-neutral).abs().max())
            speed=float(d.root_lin_vel_w.torch[0,:2].norm());yaw=abs(float(d.root_ang_vel_w.torch[0,2]))
            assert active.idle_blend>.999 and target_error<1e-4,(key,active.idle_blend,target_error)
            assert pose_rms<.13 and speed<.12 and yaw<.12,(key,pose_rms,speed,yaw)
            assert .36<float(d.root_pos_w.torch[0,2])<.45,(key,float(d.root_pos_w.torch[0,2]))
            assert max_rate<=6.001,(key,max_rate)
            recovery_results.append(dict(key=key,blend=active.idle_blend,target_error=target_error,actual_pose_rms=pose_rms,actual_speed=speed,actual_yaw=yaw,max_target_rate=max_rate))
    print('[RECOVERY_VERIFIED] '+json.dumps(recovery_results),flush=True)
    result=dict(status='passed',recovery=recovery_results,policies=list(c.adapters),terrain=terrain['kind'],parity=errors)
    (folder/'verified.json').write_text(json.dumps(result,indent=2));print('[IMPORT_VERIFIED] '+json.dumps(result),flush=True)
except BaseException:
    traceback.print_exc();sys.stderr.flush();(folder/'failed').write_text(traceback.format_exc());raise
finally:
    if env:env.close()
    app.close()
