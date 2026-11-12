"""Activate the downloaded student at the user's request, recording known instability."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
from local_policy_deployment import HANDSTAND_RECEIPT,apply_deployment
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)
def main():
    status=json.loads((HERE/'server_status.json').read_text())
    candidate=json.loads((HERE/'candidate_policy.json').read_text())
    root=Path(candidate['student_training_root']);student=Path(candidate['checkpoint'])
    assert sha(student)==status['checkpoint_sha256']
    assert sha(HERE/'student_999_export.tar.gz')==status['bundle_sha256']
    launch=json.loads((root/'distillation_launch.json').read_text())
    teacher=bootstrap.SNAPSHOT/'go2w_handstand_currentU80_20261108/logs/train80/teacher_4179.pt'
    assert sha(teacher)==launch['teacher_sha256']==candidate['teacher_sha256']
    ck=torch.load(student,map_location='cpu',weights_only=False)
    teacher_ck=torch.load(teacher,map_location='cpu',weights_only=False)
    assert ck['kind']=='student' and ck['iteration']==999
    assert ck['interface']==teacher_ck['interface']==launch['source_interface']
    for name,digest in ck['interface']['sources'].items():assert sha(root/'go2w_mimic'/name)==digest,name
    assert sha(root/'go2w_mimic/learning.py')==launch['trainer_sha256']
    assert sha(root/'distill_driver.py')==launch['driver_sha256']
    assert sha(root/'assets/go2w_description/urdf/go2w_description.urdf')==ck['interface']['urdf_sha256']
    assert sha(BASE/'runtime/handstand_student_adapter.py')==sha(HERE/'handstand_student_adapter.py')
    cfg_path=BASE/'runtime/config.json';before=json.loads(cfg_path.read_text(encoding='utf-8'))
    assert Path(before['policies']['wheel_locomotion']['checkpoint']).name=='policy_3824.pt'
    proposed=bootstrap.load_config()
    proposed['policies']['handstand'].update(json.loads(json.dumps(candidate).replace('/home/ubuntu/Go2W/',bootstrap.SNAPSHOT.as_posix()+'/')))
    bootstrap.load_config=lambda:json.loads(json.dumps(proposed))
    from run import create_runtime
    env,controller=create_runtime()
    assert controller.adapters['handstand'].ck['kind']=='student'
    assert len(controller.adapters)==7
    current=json.loads(cfg_path.read_text(encoding='utf-8'));assert current==before
    current['policies']['handstand'].update(candidate)
    backup=BASE/'backups'/('before_user_requested_student999_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir();shutil.copy2(cfg_path,backup/'config.json')
    if HANDSTAND_RECEIPT.exists():shutil.copy2(HANDSTAND_RECEIPT,backup/'deployment.json')
    receipt=dict(imported_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        policies={'handstand':candidate},checkpoint_sha256=sha(student),teacher_sha256=sha(teacher),
        training=dict(iterations=1000,num_envs=2048,teacher_iteration=4179),
        isaac_evaluation=status['evaluation'],config_backup=str(backup),
        activation='explicit_user_request_after_disclosure_of_mujoco_instability',
        verification=dict(source_and_weight_hashes=True,all_seven_policies_loaded=True,
            mujoco_stability_passed=False,known_failure='Student lost handstand and triggered original guard at H age 4.88 s in idle diagnostic',
            physics_or_guards_changed=False))
    assert {k:v for k,v in current['policies'].items() if k!='handstand'}=={k:v for k,v in before['policies'].items() if k!='handstand'}
    assert {k:v for k,v in current.items() if k!='policies'}=={k:v for k,v in before.items() if k!='policies'}
    save(HANDSTAND_RECEIPT,receipt);save(cfg_path,current)
    assert apply_deployment(json.loads(json.dumps(current)))==current
    save(HERE/'import_result.json',receipt)
    save(HERE/'background_status.json',dict(state='imported_by_user_request',active_iteration=999,
        known_mujoco_instability=True,config_backup=str(backup),updated_at=receipt['imported_at']))
    print('IMPORTED student_999; wheel_3824 retained; backup='+str(backup),flush=True)
if __name__=='__main__':main()
