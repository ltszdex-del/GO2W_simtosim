from pathlib import Path
import json,hashlib,shutil,sys
import torch
r=Path('/home/ubuntu/Go2W/interactive_stage1_20261104')
sys.path.insert(0,str(r));from commands import profile_from_contract
roots={'locomotion':Path('/home/ubuntu/Go2W/go2w_step_sagittal400_20261107'),
       'wheel_locomotion':Path('/home/ubuntu/Go2W/go2w_wheel_brake400_20261107')}
cks={}
for mode,root in roots.items():
    log=root/'logs/continue400'
    assert (log/'exit_code').read_text().strip()=='0',f'{mode}: final evaluation has not exited'
    ckpath=Path((log/'latest.txt').read_text().strip());ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['iteration']==(6699 if mode=='locomotion' else 899),ck['iteration']
    for name,h in ck['manifest']['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    cks[mode]=(ckpath,ck)
cfg=json.loads((r/'config.json').read_text())
for mode,(ckpath,ck) in cks.items():
    cfg['policies'][mode].update(checkpoint=str(ckpath),training_root=str(roots[mode]))
step=cks['locomotion'][1]['manifest'];wheel=cks['wheel_locomotion'][1]['manifest']
cfg.update(profile_from_contract(step['command'],step['control']))
cfg['wheel_command_profile']=profile_from_contract(wheel['command'],wheel['control'])
cfg['wheel_drive_target_rate_rad_s2']=wheel['control']['wheel_accel']
(r/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(dict(checkpoints={m:str(x[0]) for m,x in cks.items()},step=profile_from_contract(step['command'],step['control']),wheel=cfg['wheel_command_profile']),indent=2))
