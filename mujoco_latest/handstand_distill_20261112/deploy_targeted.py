"""Activate the targeted student only after paired physics and six MuJoCo checks."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
from local_policy_deployment import HANDSTAND_RECEIPT,apply_deployment
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(temporary,path)
root=BASE/'policies/handstand_targeted_20261112';checkpoint=root/'logs/targeted1000/student_999.pt'
assert sha(checkpoint)=='7301769ab21e881357f4ab1c81e8778b2eed21277a43426c982455af639ac1a7'
launch=json.loads((root/'targeted_launch.json').read_text());assert sha(root/'targeted_driver.py')==launch['driver_sha256']
rows=[json.loads(s) for s in (HERE/'targeted_progress_snapshot.jsonl').read_text().splitlines()]
assert [r['iteration'] for r in rows]==list(range(1000))
checks=[]
for label in ['idle','W','Q','E','E_turn150','wheel']:
    report=BASE/'logs'/('handstand_targeted999_'+label+'.json');result=json.loads(report.read_text())
    assert result['passed'] and len(result['held'])==2
    assert max(result['standing_leg_rms'])<.1
    checks.append(dict(case=label,report=str(report),sha256=sha(report),standing_leg_rms=result['standing_leg_rms']))
evaluation=json.loads((HERE/'targeted_isaac20s.json').read_text())
assert evaluation['student']['success_rate']>=.9 and evaluation['student']['survival_rate']>=.9
cfg_path=BASE/'runtime/config.json';before=json.loads(cfg_path.read_text(encoding='utf-8'))
assert Path(before['policies']['wheel_locomotion']['checkpoint']).name=='policy_3824.pt'
old=Path(before['policies']['handstand']['checkpoint'])
assert sha(old)=='5bd64542fd74f52534c465f7836b983305a46a799fee95f908334593e2382882'
previous=torch.load(old,map_location='cpu',weights_only=False);new=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert previous['interface']==new['interface'] and new['kind']=='student'
candidate=dict(factory='handstand_student_adapter:create',checkpoint=checkpoint.as_posix(),student_training_root=root.as_posix())
current=json.loads(json.dumps(before));current['policies']['handstand'].update(candidate)
backup=BASE/'backups'/('before_targeted_student999_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir();shutil.copy2(cfg_path,backup/'config.json')
if HANDSTAND_RECEIPT.exists():shutil.copy2(HANDSTAND_RECEIPT,backup/'deployment.json')
receipt=dict(imported_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    policies={'handstand':candidate},checkpoint_sha256=sha(checkpoint),teacher_sha256=launch['teacher_sha256'],
    training=dict(targeted_updates=1000,num_envs=2048,teacher_iteration=4179,warm_student_sha256=sha(old)),
    isaac_evaluation20s=evaluation,mujoco_checks=checks,config_backup=str(backup),
    original_job_exit='training updates complete; automatic evaluation failed due to inference-mode reset; separate evaluation succeeded',
    separate_evaluation_report=str(HERE/'targeted_isaac20s.json'))
assert {k:v for k,v in current['policies'].items() if k!='handstand'}=={k:v for k,v in before['policies'].items() if k!='handstand'}
assert {k:v for k,v in current.items() if k!='policies'}=={k:v for k,v in before.items() if k!='policies'}
save(HANDSTAND_RECEIPT,receipt);save(cfg_path,current)
assert apply_deployment(json.loads(json.dumps(current)))==current
save(HERE/'targeted_import_result.json',receipt)
save(HERE/'background_status.json',dict(state='targeted_student_imported',checkpoint=str(checkpoint),
    sha256=sha(checkpoint),mujoco_six_cases_passed=True,backup=str(backup),updated_at=receipt['imported_at']))
print('IMPORTED_TARGETED_STUDENT999; 6 MuJoCo cases passed; wheel3824 unchanged',flush=True)
