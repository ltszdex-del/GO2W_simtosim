"""Three-axis interactive commands, explicit PD and validated policy adapters."""
import argparse
import json
import time
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher
from control import Controller
from bridge import serve

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--steps',type=int,default=0)
parser.add_argument('--config',default='config.json')
parser.add_argument('--selftest_handstand',action='store_true')
parser.add_argument('--selftest_pu',action='store_true')
parser.add_argument('--selftest_getup',action='store_true')
parser.add_argument('--log_dir',default='logs/live')
parser.add_argument('--bridge_port',type=int)
parser.add_argument('--screenshot',help='Save a server-side viewport screenshot for visual QA')
parser.add_argument('--selftest_virtual',action='store_true',help='Exercise virtual button callbacks once, then release')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
cfg=json.loads((ROOT/args.config).read_text())
if args.selftest_getup:
    source=Path(cfg['policies']['getting_up']['source_root'])
    initial=json.loads((source/'reference_states.json').read_text())[0]
    cfg['initial_standing']=initial;cfg['initial_leg_target']=initial['q']
if args.bridge_port: cfg['bridge_port']=args.bridge_port
controller=Controller(cfg)
wheel_test_profiles=set();wheel_test_modes=set()
launcher=AppLauncher(args)
app=launcher.app
env=server=subscription=None
folder=ROOT/args.log_dir
folder.mkdir(parents=True,exist_ok=True)
try:
    import torch
    from scene import StandingEnv,EnvCfg
    ec=EnvCfg(); ec.sim.device=args.device or 'cuda:0'
    if cfg.get('robot_asset_root'):
        ec.scene.robot.spawn.asset_path=str(Path(cfg['robot_asset_root'])/'urdf/go2w_description.urdf')
    if cfg.get('initial_standing'):
        initial=cfg['initial_standing'];pose=initial['root_pose']
        ec.scene.robot.init_state.pos=tuple(pose[:3]);ec.scene.robot.init_state.rot=tuple(pose[3:])
        from scene import LEGS,WHEELS
        ec.scene.robot.init_state.joint_pos=dict(zip(LEGS+WHEELS,initial['q']+initial['wheel_q']))
    ec.viewer.eye=(1.8,1.8,1.2); ec.viewer.lookat=(0.,0.,.28)
    ec.scene.robot.actuators['motors'].damping={'.*_hip_joint':0.,'.*_thigh_joint':0.,'.*_calf_joint':0.,'.*_foot_joint':100.}
    env=StandingEnv(ec,cfg);env.reset()
    wheel_damping=env.robot.data.joint_damping.torch[0,env.wheels]
    assert torch.allclose(wheel_damping,torch.full_like(wheel_damping,100.)),wheel_damping
    import carb.settings
    settings=carb.settings.get_settings()
    settings.set('/rtx/post/aa/op',4)
    settings.set('/rtx/post/dlss/execMode',2)
    from arrows import VelocityArrows
    from types import SimpleNamespace
    arrows=VelocityArrows()
    print('[PHYSICS] mass=',float(env.robot.data.body_mass.torch[0].sum()),'wheel damping=',wheel_damping.tolist(),flush=True)
    controller.contract=env.contract()
    controller.policy_contracts['wheel_locomotion']=env.contract('wheel_locomotion')
    controller.policy_contracts['handstand']=env.contract('handstand')
    controller.policy_contracts['handstand_return']=env.contract('handstand_return')
    for mode in ('lying_down','getting_up','standing_hold'):controller.policy_contracts[mode]=env.contract(mode)
    (folder/'environment_manifest.json').write_text(json.dumps(controller.contract,indent=2))
    from plugin_adapters import load_adapter
    for mode,spec in cfg['policies'].items():
        if spec is None:continue
        try:
            adapter=load_adapter(spec,ROOT,env.device)
            controller.register(mode,adapter)
            if hasattr(adapter,'bind_env'):adapter.bind_env(env)
        except Exception as error:
            controller.policy_errors[mode]=str(error)
            controller.message=f'{mode} 加载失败：{error}'
            print('[POLICY_LOAD_ERROR]',mode,repr(error),flush=True)
    if 'locomotion' not in controller.adapters:raise RuntimeError('WASD adapter failed: '+str(controller.policy_errors))
    if args.selftest_getup:
        controller.mode='prone'
        controller.feed('selftest',['U']);controller.feed('selftest',[])
    controller.message='WASD policy loaded; Q/E disabled for this checkpoint'
    server=serve(controller)
    panel=None
    try:
        from stream_ui import StreamPanel
        panel=StreamPanel(controller)
        print('[UI] stream panel created independently of native keyboard',flush=True)
    except Exception as error:
        print(f'[UI] unavailable: {error!r}',flush=True)
    try:
        import carb.input
        import omni.appwindow
        keyboard=omni.appwindow.get_default_app_window().get_keyboard()
        inputs=carb.input.acquire_input_interface()
        from keyboard_router import KeyboardRouter
        keyboard_router=KeyboardRouter(controller,lambda: panel.reopen() if panel else None)
        def keyboard_down(key):
            aliases={'W':'UP','S':'DOWN','A':'LEFT','D':'RIGHT'}
            names=(key,aliases[key]) if key in aliases else (key,)
            return any(inputs.get_keyboard_value(keyboard,getattr(carb.input.KeyboardInput,name))>0.
                       for name in names)
        # Check the binding at startup; polling failures do not disable the watchdog.
        keyboard_down('W')
        print('[KEYBOARD] held-state polling enabled; key repeat is not required',flush=True)
        def on_key(event,*_):
            if keyboard_router.handle(event):
                print('[KEY]',event.type,str(getattr(event.input,'name',event.input)),flush=True)
            return True
        subscription=inputs.subscribe_to_keyboard_events(keyboard,on_key)
        print('[KEYBOARD] native WebRTC keyboard listener enabled', flush=True)
    except Exception as error:
        print(f'[KEYBOARD] native keyboard unavailable ({type(error).__name__}); use HTTP bridge 127.0.0.1:8765', flush=True)
    (folder/'runtime.json').write_text(json.dumps(dict(config=cfg,joints=env.robot.joint_names,
        leg_order=env.legs,wheel_order=env.wheels,root_resets='initial only'),indent=2))
    print(f"[READY] policies={list(controller.adapters)}; input bridge 127.0.0.1:{cfg['bridge_port']}",flush=True)
    start=time.monotonic();step=0;handstand_peak_pitch=0.;handstand_upright_steps=0
    action=torch.zeros((1,16),device=env.device)
    pu_test_phase='hold';pu_test_events=[];pu_test_start=0
    with (folder/'trace.jsonl').open('w',encoding='utf-8') as trace,torch.inference_mode():
        while app.is_running() and (args.steps==0 or step<args.steps):
            tick=time.monotonic()
            if controller.reset_requested:
                env.handstand_context=None
                env.reset()
                env.target=torch.tensor(cfg['initial_leg_target'],device=env.device).reshape(1,12)
                env.goal=env.target.clone();env.wheel_target.zero_();env.wheel_goal.zero_();env.torque.zero_()
                env.set_motor_context(None,'reset',100.)
                env.robot.set_joint_effort_target(env.torque,joint_ids=env.ids)
                env.robot.set_joint_velocity_target(torch.zeros_like(env.wheel_target),joint_ids=env.wheels)
                controller.reset_runtime(env.snapshot())
                if subscription is not None:keyboard_router.held.clear()
                if panel:panel.deadline=0.
                print('[RESET] U standing pose restored, controller/input/history cleared',flush=True)
            if args.selftest_pu:
                if step==250:pu_test_phase='walk';controller.feed('selftest',['W'])
                if 250<=step<275:controller.feed('selftest',['W'])
                if step==275:controller.feed('selftest',[]);pu_test_phase='brake'
                if step==550:
                    pu_test_events.append(dict(step=step,phase=pu_test_phase,mode=controller.mode))
                    controller.feed('selftest',['P']);controller.feed('selftest',[]);pu_test_phase='P'
                if pu_test_phase=='P' and controller.mode=='prone':
                    pu_test_events.append(dict(step=step,phase='P_complete',mode=controller.mode))
                    controller.feed('selftest',['U']);controller.feed('selftest',[]);pu_test_phase='U'
                elif pu_test_phase=='P' and step>=950 and controller.active and hasattr(controller.active,'can_start_getup') and controller.active.can_start_getup(env.snapshot()):
                    pu_test_events.append(dict(step=step,phase='P_pose_ready_without_training_success',mode=controller.mode))
                    controller.feed('selftest',['U']);controller.feed('selftest',[]);pu_test_phase='U'
                if pu_test_phase=='U' and controller.mode=='standing_hold':
                    pu_test_events.append(dict(step=step,phase='U_complete',mode=controller.mode));pu_test_phase='done'
            if args.selftest_virtual:
                if panel is None: raise RuntimeError('Virtual keyboard test requires stream panel')
                test_keys={10:'W',25:'S',40:'A',55:'D',70:'Q',85:'E',100:'H',115:'P',130:'U'}
                if step in test_keys: panel.press(test_keys[step])
                if step==150: panel.release()
                if step==160:
                    result=controller.status()
                    if not all(result['stats'].get('trigger_'+k)==1 for k in 'HPU'): raise RuntimeError('Virtual action test failed')
                    if result['command_m_s_rad_s']!=[0.,0.,0.]: raise RuntimeError('Virtual release test failed')
                    (folder/'virtual_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                    print('[VIRTUAL_TEST] button callbacks W/S/A/D/Q/E/H/P/U and release passed',flush=True)
            if panel: panel.update()
            if subscription is not None:keyboard_router.poll(keyboard_down)
            if args.selftest_handstand and step==25:
                controller.feed('selftest',['H']);controller.feed('selftest',[])
            if step in (100,600):
                controller.feed('selftest',['M']);controller.feed('selftest',[])
            if 125<=step<225:controller.feed('selftest',['W'])
            if 300<=step<400:controller.feed('selftest',['Q'])
            if step in (225,400):controller.feed('selftest',[])
            wheel_test_profiles.add(controller.mobility_profile)
            wheel_test_modes.add((controller.mobility_profile,controller.mode))
            state=env.snapshot()
            controller.tick(env.step_dt,state)
            if controller.reset_requested:continue
            if controller.pending=='handstand' and controller.active is None:
                env.set_motor_targets(controller.adapters['handstand'].preparation_targets())
            state['command']=controller.status()
            try:
                motor_command=controller.execute(state,env.step_dt)
                if motor_command is not None:env.set_motor_targets(motor_command)
            except Exception as error:
                controller.mode='error';controller.active=None
                controller.message=f'Policy execution failed: {error}'
                controller.release(controller.source);env.safe_stop()
                print('[POLICY_ERROR]',repr(error),flush=True)
            env.step(action)
            live=env.snapshot()
            if controller.mode=='handstand':
                g=live['projected_gravity_b'][0]
                pitch=float(torch.atan2(g[0],-g[2]))
                handstand_peak_pitch=max(handstand_peak_pitch,pitch)
                handstand_upright_steps+=int(pitch>1.3)
            arrows.update(SimpleNamespace(_physical={'yaw':torch.tensor([live['yaw_rad']])},
                command=SimpleNamespace(velocity=torch.tensor([controller.command])),robot=env.robot))
            completed=controller.after_step(live)
            if completed=='standing':env.standing_target()
            if completed=='error':env.safe_stop()
            if env.pd_updates!=4: raise RuntimeError(f'PD substeps {env.pd_updates}')
            step+=1
            if step%10==0:
                d=env.robot.data
                xyz=d.root_pos_w.torch[0].tolist();gravity=d.projected_gravity_b.torch[0].tolist()
                if not torch.isfinite(d.joint_pos.torch).all(): raise RuntimeError('Nonfinite robot state')
                if controller.mode in ('standing','locomotion','standing_hold') and (xyz[2]<.18 or gravity[2]>-.5):
                    controller.mode='error';controller.message='Robot fell; manual restart required'
                    controller.active=None;env.safe_stop()
                    controller.release(controller.source)
                controller.physics=dict(step=step,sim_s=round(step*env.step_dt,3),wall_s=round(time.monotonic()-start,3),
                    xyz=xyz,gravity_b=gravity,peak_torque_nm=float(env.torque.abs().max()),pd_substeps=env.pd_updates,
                    peak_contact_n=float(env.contacts.data.net_forces_w_history.torch.norm(dim=-1).max()))
                if controller.active and hasattr(controller.active,'ctx') and hasattr(controller.active.ctx,'pu_task'):
                    ctx=controller.active.ctx;m=ctx.p
                    controller.physics['pu']=dict(task=ctx.pu_task,clock_s=float(ctx.episode_length_buf[0])*.02,
                        training_success=bool(ctx.success[0]),motor_gain=float(ctx.gain[0]),
                        belly_gap_m=float(m['belly'][0]),joint_rms_rad=float(m['qe'][0].sqrt()),
                        angular_speed_rad_s=float(m['w'][0]),hold_s=float(ctx.hold[0]))
                status=controller.status()
                trace.write(json.dumps(status,ensure_ascii=False)+'\n');trace.flush()
                (folder/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
            if args.screenshot and step==200:
                import omni.kit.renderer_capture
                capture=omni.kit.renderer_capture.acquire_renderer_capture_interface()
                capture.capture_next_frame_swapchain(args.screenshot)
            if args.livestream:time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
        assert not controller.policy_errors,controller.policy_errors
        assert wheel_test_profiles=={'stepping','wheel'},wheel_test_profiles
        assert ('wheel','locomotion') in wheel_test_modes,wheel_test_modes
        assert controller.mobility_profile=='stepping' and controller.mode!='error',controller.status()
        print('[WHEEL_SWITCH_TEST] M -> wheel W/Q -> stop -> M -> stepping passed',flush=True)
        if args.selftest_getup:
            result=controller.status()
            (folder/'getup_test.json').write_text(json.dumps(result,indent=2))
            assert result['mode']=='standing_hold',result['message']
            print('[GETUP_TEST] prone -> U -> standing_hold passed',flush=True)
        if args.selftest_pu:
            (folder/'pu_test.json').write_text(json.dumps(dict(phase=pu_test_phase,events=pu_test_events,status=controller.status()),indent=2))
            print('[PU_TEST]',pu_test_phase,controller.mode,controller.message,flush=True)
        if args.selftest_handstand:
            result=controller.status()
            result['handstand_seconds']=float(controller.adapters['handstand'].ctx.episode_length_buf[0])*.02 if hasattr(controller.adapters['handstand'],'ctx') else 0.
            result['handstand_peak_pitch']=handstand_peak_pitch
            result['handstand_upright_seconds']=handstand_upright_steps*.02
            (folder/'handstand_test.json').write_text(json.dumps(result,indent=2))
            assert result['handstand_seconds']>4. and handstand_peak_pitch>1.3 and handstand_upright_steps>=50,result
except BaseException:
    traceback.print_exc()
    raise
finally:
    if subscription is not None: inputs.unsubscribe_to_keyboard_events(keyboard,subscription)
    if server: server.shutdown()
    if env: env.close()
    app.close()
