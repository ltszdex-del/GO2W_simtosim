from pathlib import Path
import json,subprocess,os
r=Path('/home/ubuntu/Go2W/interactive_stage1_20261104')
checks=sorted(r.glob('logs/skatepark_verify_*/validation.json'))
assert checks and json.loads(checks[-1].read_text())['status']=='passed'
p=r/'config.json';cfg=json.loads(p.read_text());before=dict(cfg)
cfg['terrain']=dict(kind='skatepark',usd_path='/home/ubuntu/Go2W/scenes/go2w_skatepark_20261107/go2w_skatepark.usd')
assert {k:v for k,v in cfg.items() if k!='terrain'}=={k:v for k,v in before.items() if k!='terrain'}
p.write_text(json.dumps(cfg,indent=2)+'\n')
folder=r/'logs/skatepark_main_verify_20261107';folder.mkdir(exist_ok=False)
cmd=['/home/ubuntu/envs/isaac62/bin/python','-u','main.py','--steps','30','--livestream','0','--bridge_port','18769','--device','cuda:0','--kit_args=--no-window','--log_dir',str(folder)]
with (folder/'console.log').open('w') as out:
 q=subprocess.Popen(cmd,cwd=r,env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'pid').write_text(str(q.pid));print('DEFAULT_TERRAIN=skatepark',q.pid,folder)
