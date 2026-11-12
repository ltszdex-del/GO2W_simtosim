from pathlib import Path
import datetime,json,os,shutil,subprocess,sys
h=Path(__file__).resolve().parent;live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
for source,target in [('wheel_mirror_brake_adapter.py','wheel_rear_idle_candidate.py'),('idle_recovery.py','idle_recovery.py'),('verify_rear_idle.py','verify_rear_idle.py')]:
    shutil.copy2(h/'runtime'/source,live/target)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=h/('idle_physics_'+stamp);folder.mkdir()
with (folder/'console.log').open('w') as output:
    p=subprocess.Popen([sys.executable,'-u','verify_rear_idle.py','--livestream','0','--device','cuda:0','--log_dir',str(folder)],cwd=live,
        env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=p.pid,folder=str(folder));(h/'idle_check_launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
