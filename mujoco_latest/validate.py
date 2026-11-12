"""Independent fresh-state scenarios; report failures without hiding them."""
import bootstrap
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CASES=[(False,k) for k in ('idle','W','S','A','D','Q','E','WQ','P','U','H','HH')]+[(True,k) for k in ('idle','W','S','Q','E','WQ')]
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')

def case(item):
    wheel,key=item;name=('wheel_' if wheel else 'step_')+key
    steps=800 if key in ('P','U','H','HH') else 500
    command=[sys.executable,str(bootstrap.ROOT/'run.py'),'--headless','--steps',str(steps),'--scenario',key,'--log','validation/'+name]
    if wheel:command.append('--wheel')
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    folder=bootstrap.ROOT/'logs/validation'/name
    folder.mkdir(parents=True,exist_ok=True);(folder/'console.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        row=dict(name=name,passed=False,error=(result.stderr or result.stdout)[-2000:])
    else:
        data=json.loads((folder/'result.json').read_text(encoding='utf-8'));samples=data['samples']
        active=[s for s in samples if 180<=s['step']<=240]
        target_ok=(key!='P' or data['status']['mode']=='prone') and (key not in ('U','HH') or data['status']['mode']=='standing_hold')
        if key=='H':target_ok=data['status']['mode']=='handstand' and samples[-1]['gravity'][0]>.9
        row=dict(name=name,passed=data['status']['mode']!='error' and data['steps']==steps and target_ok,
            steps=data['steps'],final_mode=data['status']['mode'],message=data['status']['message'],
            min_height=min(s['height'] for s in samples),peak_torque=max(s['torque'] for s in samples),
            mean_active_actual=[sum(s['actual'][i] for s in active)/len(active) for i in range(3)] if active else None,
            mean_active_reference=[sum(s['command'][i] for s in active)/len(active) for i in range(3)] if active else None,
            final_actual=samples[-1]['actual'])
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return row

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as executor:rows=list(executor.map(case,CASES))
    path=bootstrap.ROOT/'logs/validation/summary.json'
    path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',sum(row['passed'] for row in rows),'/',len(rows),flush=True)
