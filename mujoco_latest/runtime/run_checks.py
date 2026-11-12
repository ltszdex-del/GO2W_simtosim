from pathlib import Path
import subprocess,os,json,time
r=Path('/home/ubuntu/Go2W/interactive_stage1_20261104')
subprocess.run(['/home/ubuntu/envs/isaac62/bin/python','-m','py_compile',str(r/'main.py'),str(r/'terrain_setup.py'),str(r/'verify_terrain_contacts.py')],check=True)
folder=r/('logs/skatepark_verify_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));folder.mkdir(exist_ok=False)
cmd=['/home/ubuntu/envs/isaac62/bin/python','-u','verify_terrain_contacts.py','--log_dir',str(folder),'--livestream','0','--device','cuda:0','--kit_args=--no-window']
with (folder/'console.log').open('w') as out:
 q=subprocess.Popen(cmd,cwd=r,env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES'),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(folder/'pid').write_text(str(q.pid));print(q.pid,folder)
