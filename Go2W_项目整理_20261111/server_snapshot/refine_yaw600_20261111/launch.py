from pathlib import Path
import subprocess,sys,json
h=Path('/home/ubuntu/Go2W/refine_yaw600_20261111')
for kind in ('wheel','step'):
    assert not (h/f'{kind}_status.json').exists(),'Already launched'
    with (h/f'{kind}_supervisor.log').open('w') as output:
        p=subprocess.Popen([sys.executable,'-u',str(h/'queue_one.py'),kind],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(kind=kind,supervisor_pid=p.pid)))
