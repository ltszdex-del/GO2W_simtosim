from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,sys,time,torch
b=Path('/home/ubuntu/Go2W');h=Path(__file__).resolve().parent;live=b/'interactive_latest_test_20261110'
sys.path.insert(0,str(live))
from commands import profile_from_contract
cfg=json.loads((live/'config.json').read_text());checkpoints={}
for kind,mode in [('step','locomotion'),('wheel','wheel_locomotion')]:
    root=b/('go2w_wheel_accel200_20261111' if kind=='wheel' else 'go2w_step_rearfix600_20261111');path=Path((root/'logs/train/latest.txt').read_text().strip())
    ck=torch.load(path,map_location='cpu',weights_only=False);m=ck['manifest']
    for name,digest in m['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((root/'assets/go2w_description/urdf/go2w_description.urdf').read_bytes()).hexdigest()==m['urdf_sha256']
    profile=profile_from_contract(m['command'],m['control']);profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
    if kind=='step':cfg.update(profile)
    else:cfg['wheel_command_profile']=profile
    cfg['policies'][mode].update(checkpoint=str(path),training_root=str(root));checkpoints[kind]=dict(checkpoint=str(path),iteration=ck['iteration'],profile=profile)
assert cfg['max_yaw_rad_s']==3.14 and cfg['wheel_command_profile']['max_yaw_rad_s']==4.
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=live/'backups'/('deploy_rearfix_'+stamp);backup.mkdir(parents=True)
for name in ('config.json','latest_deployment.json'):
    if (live/name).exists():shutil.copy2(live/name,backup/name)
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live.resolve() and any(x==b'main.py' or x==str(live/'main.py').encode() for x in args):
        os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline;time.sleep(.2)
(live/'config.json').write_text(json.dumps(cfg,indent=2))
folder=live/'logs'/('import_rearfix_'+stamp);folder.mkdir()
with (folder/'console.log').open('w') as output:
    p=subprocess.Popen([sys.executable,'-u','verify_import.py','--livestream','0','--device','cuda:0','--log_dir',str(folder)],cwd=live,
        env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(checkpoints=checkpoints,backup=str(backup),verification_pid=p.pid,verification_folder=str(folder))
(h/'latest_deployment.json').write_text(json.dumps(receipt,indent=2));(live/'latest_deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
