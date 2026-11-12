"""Stage student, verify source identity and continuous H transitions, then activate."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
from local_policy_deployment import HANDSTAND_RECEIPT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def deploy(bundle,status):
    assert status['state']=='complete' and status['iterations']==1000
    assert status['evaluation']['passed'],'Student did not pass paired Isaac evaluation'
    assert sha(bundle)==status['bundle_sha256']
    destination=HANDSTAND_RECEIPT.parent;destination.mkdir(exist_ok=True)
    root=destination/'go2w_handstand_distill4179_20261112'
    parent=bootstrap.SNAPSHOT/'go2w_handstand_currentU80_20261108'
    if root.exists():raise RuntimeError('Refusing to overwrite staged student root')
    shutil.copytree(parent,root,ignore=shutil.ignore_patterns('logs','__pycache__','.git'))
    with tarfile.open(bundle,'r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or not (root/member.name).resolve().is_relative_to(root.resolve()):
                raise ValueError('Invalid export member '+member.name)
        archive.extractall(root)
    checkpoint=root/'logs/distill1000/student_999.pt'
    assert sha(checkpoint)==status['checkpoint_sha256']
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    launch=json.loads((root/'distillation_launch.json').read_text())
    teacher=parent/'logs/train80/teacher_4179.pt'
    assert sha(teacher)==launch['teacher_sha256']=='6a8978ed66ebd850072a689d6f57694946e818fd812fb835c8a478be30d65eaa'
    teacher_ck=torch.load(teacher,map_location='cpu',weights_only=False)
    assert ck['kind']=='student' and ck['iteration']==999
    assert ck['interface']==teacher_ck['interface']==launch['source_interface']
    for name,digest in ck['interface']['sources'].items():assert sha(root/'go2w_mimic'/name)==digest,name
    assert sha(root/'go2w_mimic/learning.py')==launch['trainer_sha256']
    assert sha(root/'distill_driver.py')==launch['driver_sha256']
    assert sha(root/'assets/go2w_description/urdf/go2w_description.urdf')==ck['interface']['urdf_sha256']
    cfg_path=BASE/'runtime/config.json'
    before=json.loads(cfg_path.read_text(encoding='utf-8'))
    original_teacher=before['policies']['handstand']
    assert original_teacher['factory']=='handstand_adapter:create','Current H policy changed since distillation started'
    candidate=dict(factory='handstand_student_adapter:create',checkpoint=checkpoint.as_posix(),
        teacher_checkpoint=original_teacher['checkpoint'],teacher_sha256=sha(teacher),
        student_training_root=root.as_posix())
    candidate_path=HERE/'candidate_policy.json'
    candidate_path.write_text(json.dumps(candidate,indent=2),encoding='utf-8')
    checks=[]
    flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    for case,approach in [('idle',50),('W',50),('Q',50),('E',50),('E',150),('wheel',50)]:
        label=case+('' if approach==50 else '_turn'+str(approach))
        with (HERE/('validation_'+label+'.log')).open('w',encoding='utf-8') as output:
            subprocess.run([sys.executable,str(HERE/'validate_candidate.py'),'--candidate',str(candidate_path),
                '--case',case,'--approach-frames',str(approach)],cwd=BASE,check=True,timeout=240,
                stdout=output,stderr=subprocess.STDOUT,**flags)
        report=BASE/'logs'/('handstand_student999_'+label+'.json')
        result=json.loads(report.read_text());assert result['passed']
        checks.append(dict(case=label,report=str(report),sha256=sha(report),standing_leg_rms=result['standing_leg_rms']))
    # Read current config after validation so a completed wheel import is retained.
    current=json.loads(cfg_path.read_text(encoding='utf-8'))
    assert current['policies']['handstand']==original_teacher,'H configuration changed during validation'
    other_policies={k:v for k,v in current['policies'].items() if k!='handstand'}
    current['policies']['handstand'].update(candidate)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=BASE/'backups'/('before_student999_import_'+stamp);backup.mkdir()
    shutil.copy2(cfg_path,backup/'config.json')
    if HANDSTAND_RECEIPT.exists():shutil.copy2(HANDSTAND_RECEIPT,backup/'deployment.json')
    receipt=dict(imported_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        policies={'handstand':candidate},checkpoint_sha256=sha(checkpoint),teacher_sha256=sha(teacher),
        training=dict(iterations=1000,num_envs=2048,teacher_iteration=4179),
        isaac_evaluation=status['evaluation'],mujoco_checks=checks,config_backup=str(backup))
    temporary=HANDSTAND_RECEIPT.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2),encoding='utf-8');os.replace(temporary,HANDSTAND_RECEIPT)
    temporary=cfg_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current,indent=2),encoding='utf-8');os.replace(temporary,cfg_path)
    assert all(current['policies'][k]==v for k,v in other_policies.items())
    (HERE/'import_result.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('IMPORTED handstand student_999 after six continuous MuJoCo checks',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--status',type=Path,required=True)
    a=p.parse_args();deploy(a.bundle,json.loads(a.status.read_text(encoding='utf-8')))
