from pathlib import Path
import json,subprocess,sys
h=Path(__file__).resolve().parent
assert not (h/'launch.json').exists()
with (h/'supervisor.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(h/'run.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=p.pid);(h/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
