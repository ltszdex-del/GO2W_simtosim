from pathlib import Path
import json,torch,sys,subprocess,datetime,shutil,os,signal,time,hashlib
b=Path('/home/ubuntu/Go2W');h=b/'refine_followup_20261111';live=b/'interactive_latest_test_20261110'
sys.path.insert(0,str(live))
from commands import profile_from_contract
adapter=(live/'step_geometry_adapter.py').read_text()
needle="        assert m['phase']['period'] == .3"
assert needle in adapter
adapter=adapter.replace(needle,"        assert m['phase']['period'] == .28 and m['phase']['period_range_s']==[.23,.33]")
assert 'phase = self.age * (2*math.pi/.3)' in adapter
adapter=adapter.replace('phase = self.age * (2*math.pi/.3)',"phase = self.age * (2*math.pi/self.ck['manifest']['phase']['period'])")
(h/'step_followup_adapter.py').write_text(adapter)
cfg=json.loads((live/'config.json').read_text());checkpoints={}
for kind,mode in [('step','locomotion'),('wheel','wheel_locomotion')]:
    status=json.loads((h/f'{kind}_status.json').read_text());assert status['state']=='complete'
    path=Path(status['checkpoint']);ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['iteration']==status['goal_iteration']
    subprocess.run([sys.executable,str(h/'verify_latest_parity.py'),kind],check=True)
    profile=profile_from_contract(ck['manifest']['command'],ck['manifest']['control'])
    profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
    if kind=='step':cfg.update(profile)
    else:cfg['wheel_command_profile']=profile
    cfg['policies'][mode].update(checkpoint=str(path),training_root=str(b/f'go2w_{kind}_followup_20261111'))
    checkpoints[mode]=str(path)
cfg['policies']['locomotion']['factory']='step_followup_adapter:create'
assert cfg['max_yaw_rad_s']==1.6 and cfg['wheel_command_profile']['max_yaw_rad_s']==2.
assert cfg['wheel_command_profile']['max_forward_m_s']==3. and cfg['wheel_command_profile']['yaw_brake_rad_s2']==9.6
assert cfg['wheel_idle_assist'] and cfg['wheel_policy_handles_stopping']
verifier=(live/'verify_import.py').read_text()
needle='gait_period=torch.tensor([.30],device=env.device)'
assert needle in verifier
verifier=verifier.replace(needle,"gait_period=torch.tensor([active.ck['manifest']['phase']['period']],device=env.device)")
compile(verifier,'verify_import.py','exec')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=live/'backups'/('before_followup_'+stamp);backup.mkdir(parents=True)
for name in ['config.json','latest_deployment.json','step_followup_adapter.py','verify_import.py']:
    if (live/name).exists():shutil.copy2(live/name,backup/name)
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live and any(arg==b'main.py' or arg.endswith(b'/main.py') for arg in args):
        os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists():
        if Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]=='Z':break
        assert time.monotonic()<deadline;time.sleep(.2)
shutil.copy2(h/'step_followup_adapter.py',live/'step_followup_adapter.py')
(live/'verify_import.py').write_text(verifier)
temp=live/'config.json.tmp';temp.write_text(json.dumps(cfg,indent=2));temp.replace(live/'config.json')
folder=live/'logs'/('import_followup_'+stamp);folder.mkdir()
cmd=[sys.executable,'-u','verify_import.py','--livestream','0','--device','cuda:0','--log_dir',str(folder)]
with (folder/'console.log').open('w') as output:
    p=subprocess.Popen(cmd,cwd=live,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
        stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(root=str(live),backup=str(backup),checkpoints=checkpoints,policies=cfg['policies'],import_check_pid=p.pid,import_check_folder=str(folder),
    config_sha256=hashlib.sha256((live/'config.json').read_bytes()).hexdigest())
(h/'latest_test_deployment.json').write_text(json.dumps(receipt,indent=2));(live/'latest_deployment.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(checkpoints=checkpoints,backup=str(backup),import_check_pid=p.pid,import_check_folder=str(folder))))
