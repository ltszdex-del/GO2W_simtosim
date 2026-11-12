from pathlib import Path
import json,math,sys,random
from types import SimpleNamespace as NS
h=Path(__file__).resolve().parent;live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
sys.path.insert(0,str(live));sys.path.insert(0,str(h/'runtime'))
from commands import CommandIntegrator,profile_from_contract
from control import Controller
from keyboard_router import KeyboardRouter
import torch
results=[]
for kind in ('wheel','step'):
    # Isolated worker avoids training config module names leaking between kinds.
    import subprocess
    result=subprocess.run([sys.executable,str(h/'check_parity.py'),kind],check=True,capture_output=True,text=True)
    results.append(json.loads(result.stdout))
cfg=json.loads((live/'config.json').read_text())
for mode in ('locomotion','wheel_locomotion'):
    ck=torch.load(cfg['policies'][mode]['checkpoint'],map_location='cpu',weights_only=False)
    profile=profile_from_contract(ck['manifest']['command'],ck['manifest']['control'])
    profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
    if mode=='locomotion':cfg.update(profile)
    else:cfg['wheel_command_profile']=profile
controller=Controller(cfg);controller.adapters={'locomotion':NS(),'wheel_locomotion':NS(command_profile=cfg['wheel_command_profile'])}
controller._enter_standing_hold=lambda state:None
# Freeze locomotion selection; this test owns command integration, not robot actions.
controller.mode='prone'
router=KeyboardRouter(controller,lambda:None)
event=lambda kind:NS(input=NS(name='K'),type=NS(name=kind))
router.handle(event('KEY_PRESS'));router.handle(event('KEY_REPEAT'));router.handle(event('KEY_PRESS'))
controller.tick(.02,{'yaw_rad':0.})
assert controller.status()['speed_mode']=='LOW' and controller.stats['trigger_K']==1
for mode,forward,yaw in [('stepping',1.15,.8),('wheel',1.5,1.)]:
    controller._set_mobility(mode,{'yaw_rad':0.})
    for _ in range(500):controller.integrator.step({'W','Q'},.02)
    assert math.isclose(controller.command[0],forward,abs_tol=1e-8) and math.isclose(controller.command[2],yaw,abs_tol=1e-8)
    assert controller.integrator.cfg['yaw_accel_rad_s2']==controller.status()['command_limits']['yaw_accel_rad_s2']
router.handle(event('KEY_RELEASE'));router.handle(event('KEY_PRESS'));controller.mode='prone';controller.tick(.02)
assert controller.status()['speed_mode']=='HIGH' and controller.stats['trigger_K']==2
result=dict(passed=True,parity=results,k_repeat_debounce=True,mode_switch_preserves_low=True,acceleration_unchanged=True)
(h/'runtime_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
