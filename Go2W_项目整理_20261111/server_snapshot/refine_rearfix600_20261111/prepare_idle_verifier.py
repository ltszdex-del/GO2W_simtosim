from pathlib import Path
h=Path(__file__).resolve().parent
p=h/'runtime/verify_import.py';s=p.read_text()
needle="cfg['wheel_idle_assist']=False;launcher="
assert needle in s
s=s.replace(needle,"cfg['policies']['wheel_locomotion']['factory']='wheel_rear_idle_candidate:create';cfg['wheel_idle_assist']=False;launcher=")
start=s.index('    with torch.inference_mode():\n        env.reset();c.reset_runtime')
end=s.index("    result=dict(status='passed'",start)
s=s[:start]+'''    recovery_results=[]
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
                assert c.mode!='error' and env.robot.data.root_pos_w.torch[0,2]>.27,(key,k)
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
'''+s[end:]
s=s.replace("result=dict(status='passed',policies=", "result=dict(status='passed',recovery=recovery_results,policies=")
(h/'runtime/verify_rear_idle.py').write_text(s)
