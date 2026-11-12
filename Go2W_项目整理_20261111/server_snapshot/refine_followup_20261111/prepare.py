from pathlib import Path
import shutil,sys,subprocess,torch
b=Path('/home/ubuntu/Go2W');h=b/'refine_followup_20261111'
for kind,iteration,updates in [('wheel',1999,400),('step',9399,200)]:
    old=b/f'go2w_{kind}_geometry500_20261110';root=b/f'go2w_{kind}_followup_20261111'
    parent=old/f'logs/train500/policy_{iteration}.pt'
    assert torch.load(parent,map_location='cpu',weights_only=False)['iteration']==iteration
    assert not root.exists(),root
    shutil.copytree(h/kind,root,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(old/'assets',root/'assets',symlinks=True)
    if kind=='wheel':shutil.copy2(old/'U_final_standing.json',root/'U_final_standing.json')
    subprocess.run([sys.executable,str(h/'migrate.py'),str(root),str(parent),str(updates)],check=True)
    print('Prepared '+str(root),flush=True)
