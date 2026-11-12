from pathlib import Path
import datetime,shutil,json,subprocess,sys,os,signal,time
live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110');h=Path(__file__).parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=live/'backups'/('idle_assist_'+stamp);backup.mkdir(parents=True)
for name in ('stream_ui.py','wheel_mirror_brake_adapter.py','config.json','verify_import.py'):
    shutil.copy2(live/name,backup/name)
s=(live/'stream_ui.py').read_text()
needle="        error=s['heading_error_rad'];error_text=f'{error:+.3f}' if error is not None else '-'"
assert needle in s
s=s.replace(needle,"        actual_yaw_text=f'{actual[2]:+.3f} rad/s' if actual is not None else 'waiting'\n"+needle)
needle='            f"ACTUAL vx/vy/wz (sim): {actual_text}\\n"'
assert needle in s
s=s.replace(needle,needle+'\n            f"ACTUAL YAW RATE: {actual_yaw_text}\\n"')
s=s.replace('word_wrap=True,height=270','word_wrap=True,height=292')
ui=s
s=(live/'wheel_mirror_brake_adapter.py').read_text()
needle="        self.command=command;return self.history.flatten(1)"
assert needle in s
s=s.replace(needle,"""        self.command=command
        keys=state['command'].get('keys',[])
        d=self.env.robot.data
        actual_speed=float(d.root_lin_vel_w.torch[0,:2].norm())
        actual_yaw=abs(float(d.root_ang_vel_w.torch[0,2]))
        self.idle_requested=(not any(k in keys for k in 'WSQE') and math.hypot(*command[:2])<.06 and abs(command[2])<.08 and actual_speed<.12 and actual_yaw<.12)
        return self.history.flatten(1)""")
needle="        self.limits=e.robot.data.soft_joint_pos_limits.torch[:,e.legs];e.set_motor_context(self,'wheel_drive',3.)"
assert needle in s
s=s.replace(needle,"        self.idle_blend=0.;self.idle_requested=False\n"+needle)
needle="        self.target+=(desired-self.target).clamp(-c['leg_rate']*dt,c['leg_rate']*dt)\n        wheel=c['wheel_action_scale']*action[:,12:]"
assert needle in s
s=s.replace(needle,"""        enabled=self.env.control.get('wheel_idle_assist',False)
        goal=1. if enabled and self.idle_requested else 0.
        self.idle_blend=max(0.,min(1.,self.idle_blend+(dt/.8 if goal else -dt/.2)))
        neutral=torch.tensor(self.env.control['initial_leg_target'],device=self.device)[None]
        desired=torch.lerp(desired,neutral,self.idle_blend).clamp(self.limits[...,0],self.limits[...,1])
        self.target+=(desired-self.target).clamp(-c['leg_rate']*dt,c['leg_rate']*dt)
        wheel=c['wheel_action_scale']*action[:,12:]*(1.-self.idle_blend)""")
compile(ui,'stream_ui.py','exec');compile(s,'wheel_mirror_brake_adapter.py','exec')
# Check the actual patched adapter apply/reset methods without requiring Kit.
import ast,torch
from types import SimpleNamespace as NS
tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WheelAdapter')
apply=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
scope={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[apply],type_ignores=[])),'idle_apply','exec'),scope)
fake=NS(c=dict(leg_rate=6.,wheel_action_scale=30.1,wheel_accel=60.,wheel_brake_accel=100.),stand=torch.zeros(1,12),scale=torch.ones(1,12),limits=torch.tensor([[[-3.,3.]]*12]),target=torch.ones(1,12),wheel_target=torch.ones(1,4),idle_blend=0.,idle_requested=True,device='cpu',age=0.,env=NS(control={'wheel_idle_assist':True,'initial_leg_target':[0.]*12}))
for _ in range(40):scope['apply'](fake,torch.ones(1,16),.02)
assert fake.idle_blend>.999 and float(fake.target.abs().max())<1e-5 and float(fake.wheel_target.abs().max())<1e-5
fake.idle_requested=False
for _ in range(10):scope['apply'](fake,torch.ones(1,16),.02)
assert fake.idle_blend<1e-5
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live and any(x==b'main.py' or x.endswith(b'/main.py') for x in args):os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline;time.sleep(.2)
(live/'stream_ui.py').write_text(ui);(live/'wheel_mirror_brake_adapter.py').write_text(s)
cfg=json.loads((live/'config.json').read_text());cfg['wheel_idle_assist']=True
(live/'config.json').write_text(json.dumps(cfg,indent=2))
# Raw training-target parity is tested with the optional runtime assist disabled;
# a separate physical check verifies the assist after the parity pass.
verifier=(live/'verify_import.py').read_text()
needle="cfg=json.loads((r/'config.json').read_text());launcher=AppLauncher(a);app=launcher.app;env=None"
assert needle in verifier
verifier=verifier.replace(needle,"cfg=json.loads((r/'config.json').read_text());cfg['wheel_idle_assist']=False;launcher=AppLauncher(a);app=launcher.app;env=None")
needle="    result=dict(status='passed'"
assert needle in verifier
assist="""    env.reset();c.reset_runtime(env.snapshot());c._set_mobility('wheel',env.snapshot())
    cfg['wheel_idle_assist']=True
    with torch.inference_mode():
        for k in range(200):
            now[0]+=.02;c.feed('idle-check',[]);state=env.snapshot();c.tick(.02,state);state['command']=c.status()
            motor=c.execute(state,.02)
            if motor is not None:env.set_motor_targets(motor)
            env.step(action);c.after_step(env.snapshot())
    assert c.active.idle_blend>.99,c.active.idle_blend
    neutral=torch.tensor(cfg['initial_leg_target'],device=env.device)[None]
    assert float((c.active.target-neutral).abs().max())<1e-4
    assert float(env.robot.data.root_pos_w.torch[0,2])>.3
"""
verifier=verifier.replace(needle,assist+needle)
(live/'verify_import.py').write_text(verifier)
folder=live/'logs'/('import_idle_'+stamp);folder.mkdir()
with (folder/'console.log').open('w') as output:
    p=subprocess.Popen([sys.executable,'-u','verify_import.py','--livestream','0','--device','cuda:0','--log_dir',str(folder)],cwd=live,
        env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(backup=str(backup),import_check_pid=p.pid,import_check_folder=str(folder))
(h/'runtime_update.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
