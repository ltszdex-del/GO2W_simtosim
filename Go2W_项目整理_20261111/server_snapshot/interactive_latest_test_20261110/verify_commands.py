"""Random keyboard sequences must match the actual training reference integrators."""
from pathlib import Path
import sys,types,importlib.util,json,torch,random,math,hashlib
r=Path('/home/ubuntu/Go2W/interactive_stage1_20261104');sys.path.insert(0,str(r))
from commands import CommandIntegrator,profile_from_contract,PROFILE_KEYS
cfg=json.loads((r/'config.json').read_text());results={}
for mode in ('locomotion','wheel_locomotion'):
    spec=cfg['policies'][mode];ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);m=ck['manifest']
    profile=profile_from_contract(m['command'],m['control'])
    assert all(cfg[k]==v for k,v in profile.items()) if mode=='locomotion' else cfg['wheel_command_profile']==profile
    fake=types.ModuleType('config');fake.COMMAND=m['command'];sys.modules['config']=fake
    source=Path(spec['training_root'])/'schedule.py'
    loader=importlib.util.spec_from_file_location('source_schedule',source);mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
    training=mod.Commands(1,'cpu');runtime=CommandIntegrator(dict(profile))
    training.heading[:]=.31;runtime.yaw_target=.31;random.seed(27)
    maxerr=0.;headingerr=0.
    for k in range(6000):
        if k%11==0:
            keys={x for x in 'WASDQE' if random.random()<.30}
            if mode=='wheel_locomotion':keys-=set('AD')
        axes=torch.tensor([[int('W'in keys)-int('S'in keys),int('A'in keys)-int('D'in keys),int('Q'in keys)-int('E'in keys)]],dtype=torch.float32)
        target=training.tick(axes,.02)[0];actual=runtime.step(keys,.02)
        err=float((target-torch.tensor(actual)).abs().max());maxerr=max(maxerr,err)
        dh=abs(math.atan2(math.sin(float(training.heading[0])-runtime.yaw_target),math.cos(float(training.heading[0])-runtime.yaw_target)))
        headingerr=max(headingerr,dh)
        assert err<3e-5,(mode,k,target,actual)
        assert dh<3e-4,(mode,k,dh)
    results[mode]=dict(steps=6000,max_reference_error=maxerr,max_heading_error=headingerr)
# A complete M round trip restores every command field, including asymmetric acceleration.
from control import Controller
controller=Controller(dict(cfg));initial={k:controller.cfg[k] for k in PROFILE_KEYS}
wheel=types.SimpleNamespace(command_profile=cfg['wheel_command_profile'])
step=object();controller.adapters={'locomotion':step,'wheel_locomotion':wheel}
controller._enter_standing_hold=lambda state:True
controller._set_mobility('wheel',dict(yaw_rad=0.))
assert controller.cfg['max_forward_m_s']==2.5 and controller.cfg['backward_accel_m_s2']==5.
controller._set_mobility('stepping',dict(yaw_rad=0.))
assert controller.adapters['locomotion'] is step
assert {k:controller.cfg[k] for k in PROFILE_KEYS}==initial
print(json.dumps(dict(status='passed',command_parity=results,m_round_trip='passed'),indent=2))
(r/'logs/dual400_command_verification.json').write_text(json.dumps(results,indent=2))
