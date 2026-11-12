"""Wait for one existing run, then resume its final checkpoint for 600 updates."""
from pathlib import Path
import datetime,json,os,shutil,subprocess,sys,time

kind=sys.argv[1];b=Path('/home/ubuntu/Go2W');h=b/'refine_yaw600_20261111'
previous=b/'refine_combo300_20261111'/f'{kind}_status.json'
root=b/f'go2w_{kind}_yaw600_20261111'
def record(**data):
    data.update(supervisor_pid=os.getpid(),num_envs=2048,updates=600,waiting_for=str(previous),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=h/f'{kind}_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
try:
    record(state='queued')
    while True:
        state=json.loads(previous.read_text())
        if state['state']=='complete':break
        if state['state']=='failed':raise RuntimeError('Parent training failed: '+str(state))
        if state.get('pid'):
            try:
                os.kill(state['pid'],0)
                alive=Path(f"/proc/{state['pid']}/stat").read_text().split(') ',1)[1][0]!='Z'
            except (ProcessLookupError,FileNotFoundError):alive=False
            if not alive:
                # The parent's supervisor validates logs after its child exits.
                time.sleep(5)
                fresh=json.loads(previous.read_text())
                if fresh['state']=='complete':state=fresh;break
                if fresh['state']=='failed':raise RuntimeError('Parent validation failed: '+str(fresh))
                continue
        time.sleep(10)
    record(state='preparing',parent=state['checkpoint'])
    assert not root.exists(),root
    shutil.copytree(h/kind,root,ignore=shutil.ignore_patterns('__pycache__'))
    old=b/f'go2w_{kind}_combo300_20261111'
    shutil.copytree(old/'assets',root/'assets',symlinks=True)
    if kind=='wheel':shutil.copy2(old/'U_final_standing.json',root/'U_final_standing.json')
    subprocess.run([sys.executable,str(h/'migrate.py'),str(root),state['checkpoint'],'600'],check=True)
    for script in ('check_schedule.py','check_envelope.py'):
        subprocess.run([sys.executable,str(h/script),kind,str(root)],check=True)
    subprocess.run([sys.executable,'-u',str(h/'run_one.py'),kind],check=True)
except BaseException as error:
    record(state='failed',error=repr(error));raise
