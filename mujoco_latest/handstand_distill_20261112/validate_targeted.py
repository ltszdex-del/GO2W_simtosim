"""Candidate-only targeted student validation with isolated report names."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
import bootstrap
p=argparse.ArgumentParser();p.add_argument('--case',default='idle');p.add_argument('--approach-frames',default='50');a=p.parse_args()
cfg=bootstrap.load_config();candidate=json.loads((HERE/'candidate_policy.json').read_text().replace('/home/ubuntu/Go2W/',bootstrap.SNAPSHOT.as_posix()+'/'))
candidate['checkpoint']=(BASE/'policies/handstand_targeted_20261112/logs/targeted1000/student_999.pt').as_posix()
candidate['student_training_root']=(BASE/'policies/handstand_targeted_20261112').as_posix()
assert hashlib.sha256(Path(candidate['checkpoint']).read_bytes()).hexdigest()=='7301769ab21e881357f4ab1c81e8778b2eed21277a43426c982455af639ac1a7'
cfg['policies']['handstand'].update(candidate)
bootstrap.load_config=lambda:json.loads(json.dumps(cfg))
sys.argv=[str(BASE/'check_handstand_fix.py'),'--case',a.case,'--approach-frames',a.approach_frames,'--result-tag','handstand_targeted999']
runpy.run_path(str(BASE/'check_handstand_fix.py'),run_name='__main__')
