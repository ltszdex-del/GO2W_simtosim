from pathlib import Path
import datetime,json,os,shutil,subprocess,sys,time,torch
kind=sys.argv[1];h=Path(__file__).resolve().parent;b=h.parent
goal=3324 if kind=='wheel' else 10499
root=b/f'go2w_{kind}_rearfix600_20261111'
def record(**data):
    data.update(goal_iteration=goal,planned_num_envs=2048,supervisor_pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=h/f'{kind}_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
try:
    physical=json.loads((h/'idle_check_launch.json').read_text())
    assert json.loads((Path(physical['folder'])/'verified.json').read_text())['status']=='passed'
    assert json.loads((h/'wheel/rear_stance_verified.json').read_text())['passed']
    if kind=='wheel':
        stopped=json.loads((b/'refine_yaw600_20261111/wheel_status.json').read_text())
        assert stopped['state']=='stopped_by_user';parent=Path(stopped['checkpoint'])
    else:
        record(state='waiting_parent',waiting_for=str(b/'refine_combo300_20261111/step_status.json'))
        while True:
            state=json.loads((b/'refine_combo300_20261111/step_status.json').read_text())
            if state['state']=='complete':parent=Path(state['checkpoint']);break
            if state['state']=='failed':raise RuntimeError('Parent failed: '+str(state))
            time.sleep(10)
    ck=torch.load(parent,map_location='cpu',weights_only=False)
    updates=600;assert goal==ck['iteration']+updates
    record(state='preparing',parent=str(parent),remaining_updates=updates)
    assert not root.exists(),root
    shutil.copytree(h/kind,root,ignore=shutil.ignore_patterns('__pycache__'))
    old=parent.parents[2]
    shutil.copytree(old/'assets',root/'assets',symlinks=True)
    if kind=='wheel':shutil.copy2(old/'U_final_standing.json',root/'U_final_standing.json')
    subprocess.run([sys.executable,str(h/'migrate.py'),str(root),str(parent),str(updates)],check=True)
    for script in ('check_schedule.py','check_envelope.py'):
        subprocess.run([sys.executable,str(h/script),kind,str(root)],check=True)
    if kind=='wheel':subprocess.run([sys.executable,str(h/'check_rear_stance.py'),str(root)],check=True)
    subprocess.run([sys.executable,'-u',str(h/'run_one.py'),kind],check=True)
except BaseException as error:record(state='failed',error=repr(error));raise
