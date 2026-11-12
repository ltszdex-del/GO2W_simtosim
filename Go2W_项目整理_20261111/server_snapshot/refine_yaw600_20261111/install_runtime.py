"""Deploy tested K control now; keep command envelopes tied to loaded policies."""
from pathlib import Path
import datetime,json,os,shutil,signal,sys,time,torch
h=Path(__file__).resolve().parent;live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
assert json.loads((h/'runtime_verified.json').read_text())['passed']
sys.path.insert(0,str(live));sys.path.insert(0,str(h/'runtime'))
from commands import profile_from_contract
cfg=json.loads((live/'config.json').read_text())
for mode in ('locomotion','wheel_locomotion'):
    ck=torch.load(cfg['policies'][mode]['checkpoint'],map_location='cpu',weights_only=False)
    profile=profile_from_contract(ck['manifest']['command'],ck['manifest']['control'])
    profile.update(wheel_envelope_lever=0.,wheel_reference_speed_m_s=0.)
    if mode=='locomotion':cfg.update(profile)
    else:cfg['wheel_command_profile']=profile
names=['commands.py','control.py','keyboard_router.py','stream_ui.py','controls_panel.html','yaw_envelope.py']
for name in names:
    if name.endswith('.py'):compile((h/'runtime'/name).read_text(),name,'exec')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=live/'backups'/('yaw600_k_'+stamp);backup.mkdir(parents=True)
for name in names+['config.json']:
    if (live/name).exists():shutil.copy2(live/name,backup/name)
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live and any(x==b'main.py' or x.endswith(b'/main.py') for x in args):
        os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline;time.sleep(.2)
for name in names:shutil.copy2(h/'runtime'/name,live/name)
(live/'config.json').write_text(json.dumps(cfg,indent=2))
result=dict(backup=str(backup),stopped=stopped,policy_checkpoints_unchanged=True,
    step_profile={k:cfg[k] for k in profile},wheel_profile=cfg['wheel_command_profile'])
(h/'runtime_install.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
