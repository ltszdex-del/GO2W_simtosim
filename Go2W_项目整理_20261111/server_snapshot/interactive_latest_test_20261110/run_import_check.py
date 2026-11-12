from pathlib import Path
import subprocess,os
r=Path('/home/ubuntu/Go2W/interactive_stage1_20261104');folder=r/'logs/dual400_import_check';folder.mkdir(exist_ok=True)
cmd=['/home/ubuntu/envs/isaac62/bin/python','-u','verify_import.py','--log_dir',str(folder),'--livestream','0','--device','cuda:0','--kit_args=--no-window']
with (folder/'console.log').open('w') as out:q=subprocess.Popen(cmd,cwd=r,env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'pid').write_text(str(q.pid));print(q.pid,folder)
