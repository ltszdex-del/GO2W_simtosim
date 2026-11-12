import json,time,urllib.request
from pathlib import Path
base='http://127.0.0.1:8765'
def get():
    with urllib.request.urlopen(base+'/status',timeout=5) as r:return json.load(r)
def feed(keys):
    body=json.dumps(dict(session='codex-k-validation',keys=keys)).encode()
    with urllib.request.urlopen(urllib.request.Request(base+'/input',data=body,headers={'Content-Type':'application/json'}),timeout=5) as r:return r.read()
before=get();assert not before['keys'] and before['speed_mode']=='HIGH','User input active or already LOW; do not interfere'
assert before['input_age_s']>2.,'User input recently active; do not interfere'
try:
    feed(['K']);feed([])
    time.sleep(.3);low=get();assert low['speed_mode']=='LOW',low
    for key in ('max_forward_m_s','min_forward_m_s','max_lateral_m_s','max_yaw_rad_s'):
        assert low['command_limits'][key]==before['command_limits'][key]/2,key
    for key in ('yaw_accel_rad_s2','yaw_brake_rad_s2','forward_accel_m_s2','forward_brake_m_s2'):
        assert low['command_limits'][key]==before['command_limits'][key],key
finally:
    if get()['speed_mode']=='LOW':feed(['K']);feed([]);time.sleep(.3)
after=get();assert after['speed_mode']=='HIGH' and after['policy_errors']=={}
result=dict(passed=True,live_low_limits=low['command_limits'],restored_high=True,physics_step=after['physics']['step'])
(Path(__file__).parent/'live_k_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
