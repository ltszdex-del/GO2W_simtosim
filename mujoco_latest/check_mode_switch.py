"""Real controller and MuJoCo: step -> wheel -> step -> each movement key."""
import bootstrap,json,torch
from run import create_runtime,advance
rows=[]
for key in 'WSADQE':
    env,c=create_runtime();seen_wheel=False;step_started=False
    with torch.inference_mode():
        for frame in range(550):
            keys=['M'] if frame in (50,260) else (['W'] if 100<=frame<135 else ([key] if 330<=frame<375 else []))
            c.feed('mode_switch_test',keys);s=advance(env,c)
            seen_wheel|=s['mobility_profile']=='wheel'
            if 340<=frame<375:
                assert s['mobility_profile']=='stepping',s
                assert c.active is c.step_adapter,s
                assert max(abs(v) for v in s['command_m_s_rad_s'])>.1,s
                step_started=True
            assert s['mode']!='error',(key,frame,s['message'])
    assert seen_wheel and step_started
    # Simulate the shared endpoint of U and H-return; new commands must leave it.
    c._enter_standing_hold(env.snapshot())
    for frame in range(20):
        c.feed('mode_switch_test',[key]);s=advance(env,c)
    assert c.mode=='locomotion' and c.active is c.step_adapter,s
    row=dict(key=key,passed=True,profile=c.mobility_profile,mode=c.mode)
    rows.append(row);print(row,flush=True)
(bootstrap.ROOT/'logs/mode_switch_checks.json').write_text(json.dumps(rows,indent=2))
