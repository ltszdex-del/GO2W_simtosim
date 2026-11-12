"""Run the existing continuous H regression with a staged student configuration."""
import argparse,json,runpy,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import bootstrap
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--case',required=True);p.add_argument('--approach-frames',default='50')
a=p.parse_args()
cfg=bootstrap.load_config()
candidate=json.loads(a.candidate.read_text(encoding='utf-8').replace(
    '/home/ubuntu/Go2W/',bootstrap.SNAPSHOT.as_posix()+'/'))
cfg['policies']['handstand'].update(candidate)
bootstrap.load_config=lambda:cfg
sys.argv=[str(BASE/'check_handstand_fix.py'),'--case',a.case,'--approach-frames',a.approach_frames,
    '--result-tag','handstand_student999']
runpy.run_path(str(BASE/'check_handstand_fix.py'),run_name='__main__')
