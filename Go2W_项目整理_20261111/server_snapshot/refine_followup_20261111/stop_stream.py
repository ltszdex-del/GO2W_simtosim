from pathlib import Path
import json,os,signal,time,subprocess
live=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
h=Path(__file__).parent
for kind in ('wheel','step'):
    s=json.loads((h/f'{kind}_status.json').read_text())
    assert s['state']=='running' and s['phase']=='train'
    assert (Path(s['log'])/'progress.jsonl').stat().st_size>0
    os.kill(s['pid'],0)
folder=Path((live/'logs/current.txt').read_text().strip())
receipt=json.loads((folder/'launch.json').read_text());pid=receipt['pid']
proc=Path(f'/proc/{pid}')
if proc.exists():
    assert (proc/'cwd').resolve()==live
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert any(x==b'main.py' or x.endswith(b'/main.py') for x in args)
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while (proc/'stat').exists() and (proc/'stat').read_text().split(') ',1)[1][0]!='Z':
        assert time.monotonic()<deadline,'Stream did not exit'
        time.sleep(.2)
for kind in ('wheel','step'):
    s=json.loads((h/f'{kind}_status.json').read_text());os.kill(s['pid'],0)
deadline=time.monotonic()+10
while True:
    ports=[]
    for protocol in ('tcp','tcp6'):
        for line in Path('/proc/net/'+protocol).read_text().splitlines()[1:]:
            fields=line.split()
            if int(fields[1].split(':')[1],16) in (49100,8765) and fields[3]=='0A':ports.append(line)
    if not ports or time.monotonic()>deadline:break
    time.sleep(.2)
assert not ports,ports
result=dict(stopped_stream_pid=pid,training_preserved=True,stream_ports_closed=True)
(folder/'stopped.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader'])
