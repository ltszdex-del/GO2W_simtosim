from pathlib import Path
import datetime,json,os,signal,time
b=Path('/home/ubuntu/Go2W');h=b/'refine_yaw600_20261111';all_stopped=[]
for kind in ('wheel','step'):
    path=h/f'{kind}_status.json';state=json.loads(path.read_text())
    root=b/f'go2w_{kind}_yaw600_20261111';targets=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x];cwd=(proc/'cwd').resolve()
        except (OSError,UnicodeError):continue
        worker=any(a in (str(h/'queue_one.py'),str(h/'run_one.py')) for a in args) and kind in args
        trainer=cwd==root and any(a=='main.py' or a==str(root/'main.py') for a in args)
        if worker or trainer:targets.add(int(proc.name))
    for pid in targets:
        try:os.kill(pid,signal.SIGSTOP)
        except ProcessLookupError:pass
    latest=root/'logs/train/latest.txt';checkpoint=latest.read_text().strip() if latest.exists() else str(root/'resume/parent.pt') if (root/'resume/parent.pt').exists() else None
    for pid in targets:
        try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
    end=time.monotonic()+20
    for pid in targets:
        while Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':
            if time.monotonic()>end:os.kill(pid,signal.SIGKILL);break
            time.sleep(.1)
    receipt=dict(state='stopped_by_user',reason='Wait for backward stance and runtime recovery fixes',checkpoint=checkpoint,
        previous_state=state,stopped_pids=sorted(targets),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(receipt,indent=2));all_stopped.append(dict(kind=kind,**receipt))
(h/'stopped_for_rearfix.json').write_text(json.dumps(all_stopped,indent=2));print(json.dumps(all_stopped))
