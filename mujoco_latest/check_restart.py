import bootstrap,json,torch
from run import create_runtime,advance
rows=[]
for second in ('W','S','A','D','Q','E'):
    env,c=create_runtime();modes=set();failed=None
    with torch.inference_mode():
        for step in range(600):
            keys=['W'] if 50<=step<85 else ([second] if 150<=step<190 or 270<=step<310 or 390<=step<430 else [])
            c.feed('restart_test',keys);s=advance(env,c);modes.add(s['mode'])
            if s['mode']=='error':failed=step;break
    row=dict(second=second,failed_step=failed,modes=sorted(modes),final_height=s['physics']['xyz'][2]);rows.append(row);print(row,flush=True)
(bootstrap.ROOT/'logs/restart_checks.json').write_text(json.dumps(rows,indent=2))
