from pathlib import Path
import shutil,sys,subprocess,json
b=Path('/home/ubuntu/Go2W');h=b/'refine_combo300_20261111'
for kind in ('wheel','step'):
    old=b/f'go2w_{kind}_followup_20261111';root=b/f'go2w_{kind}_combo300_20261111'
    state=json.loads((b/'refine_followup_20261111'/f'{kind}_status.json').read_text());assert state['state']=='complete'
    assert not root.exists(),root
    shutil.copytree(h/kind,root,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(old/'assets',root/'assets',symlinks=True)
    if kind=='wheel':shutil.copy2(old/'U_final_standing.json',root/'U_final_standing.json')
    subprocess.run([sys.executable,str(h/'migrate.py'),str(root),state['checkpoint'],'300'],check=True)
    subprocess.run([sys.executable,str(h/'check_schedule.py'),kind],check=True)
    print('Prepared '+str(root),flush=True)
