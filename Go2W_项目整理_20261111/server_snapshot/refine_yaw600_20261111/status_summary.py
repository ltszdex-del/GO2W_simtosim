from pathlib import Path
import json,subprocess
b=Path('/home/ubuntu/Go2W');h=Path(__file__).resolve().parent
for kind in ('wheel','step'):
    result=dict(kind=kind)
    for label,folder,origin in [('parent',b/'refine_combo300_20261111',2399 if kind=='wheel' else 9599),('new',h,2699 if kind=='wheel' else 9899)]:
        status=json.loads((folder/f'{kind}_status.json').read_text());result[label]=status
        log=Path(status.get('log','/tmp/not_present'))/'progress.jsonl'
        if log.exists():
            row=json.loads(log.read_text().splitlines()[-1]);result[label]['finished_updates']=row['iteration']-origin
            result[label]['last']={k:row[k] for k in ('iteration','loss','yaw_limit_min','yaw_limit_max','physical/failure_fraction') if k in row}
    print(json.dumps(result))
subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'])
