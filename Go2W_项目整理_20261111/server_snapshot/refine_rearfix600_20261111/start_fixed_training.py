from pathlib import Path
import json,subprocess,sys
h=Path(__file__).resolve().parent
for kind in ('wheel','step'):
    assert not (h/f'{kind}_status.json').exists(),'Already started'
    with (h/f'{kind}_supervisor.log').open('w') as output:
        p=subprocess.Popen([sys.executable,'-u',str(h/'restart_worker.py'),kind],stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(kind=kind,supervisor_pid=p.pid)))
