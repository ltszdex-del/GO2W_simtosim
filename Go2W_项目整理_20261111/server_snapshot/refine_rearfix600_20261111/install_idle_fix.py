from pathlib import Path
import datetime,json,os,shutil,signal,time
h=Path(__file__).resolve().parent;live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
check=json.loads((h/'idle_check_launch.json').read_text());verified=json.loads((Path(check['folder'])/'verified.json').read_text())
assert verified['status']=='passed' and len(verified['recovery'])==4
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=live/'backups'/('rear_idle_fix_'+stamp);backup.mkdir(parents=True)
names=['wheel_mirror_brake_adapter.py','idle_recovery.py','control.py','stream_ui.py']
for name in names:
    compile((h/'runtime'/name).read_text(),name,'exec')
    if (live/name).exists():shutil.copy2(live/name,backup/name)
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0');cwd=(proc/'cwd').resolve()
    except OSError:continue
    if cwd==live and any(x==b'main.py' or x==str(live/'main.py').encode() for x in args):
        os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
deadline=time.monotonic()+30
for pid in stopped:
    while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline;time.sleep(.2)
for name in names:shutil.copy2(h/'runtime'/name,live/name)
receipt=dict(backup=str(backup),stopped=stopped,physical_verification=check['folder'])
(h/'runtime_install.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
