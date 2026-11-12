from pathlib import Path
import json,subprocess,math
h=Path('/home/ubuntu/Go2W/refine_followup_20261111')
for kind,origin in [('wheel',1999),('step',9399)]:
    s=json.loads((h/f'{kind}_status.json').read_text());folder=Path(s.get('log','/tmp/nonexistent'))
    rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()] if (folder/'progress.jsonl').exists() else []
    row=rows[-1] if rows else {}
    print(json.dumps(dict(kind=kind,status=s,updates=row.get('iteration',origin)-origin,last={k:v for k,v in row.items() if k in ['iteration','fps','loss','gait/period_min_s','gait/period_max_s','drive_speed_cap','yaw_course_cap']},finite_losses=all(all(math.isfinite(v) for v in x['loss']) for x in rows),tail=(folder/'console.log').read_text(errors='replace').splitlines()[-2:])))
subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'])
