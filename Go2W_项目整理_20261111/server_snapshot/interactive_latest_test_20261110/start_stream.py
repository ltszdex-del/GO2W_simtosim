from pathlib import Path
import datetime,json,os,socket,subprocess,hashlib
r=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
cfg=json.loads((r/'config.json').read_text())
for spec in cfg['policies'].values():assert Path(spec['checkpoint']).is_file(),spec
for kind,port in [(socket.SOCK_STREAM,49100),(socket.SOCK_DGRAM,47998),(socket.SOCK_STREAM,8765)]:
    with socket.socket(socket.AF_INET,kind) as sock:
        if kind==socket.SOCK_STREAM:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('0.0.0.0',port))
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=r/'logs'/('test_stream_'+stamp);folder.mkdir()
cmd=['bash','start.sh','--steps','0','--log_dir',str(folder),'--screenshot',str(folder/'preview.png')]
with (folder/'console.log').open('w') as log:
    proc=subprocess.Popen(cmd,cwd=r,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,folder=str(folder),policies={k:v['checkpoint'] for k,v in cfg['policies'].items()},
    config_sha256=hashlib.sha256((r/'config.json').read_bytes()).hexdigest())
(folder/'launch.json').write_text(json.dumps(receipt,indent=2));(folder/'pid').write_text(str(proc.pid))
(r/'logs/current.txt').write_text(str(folder)+'\n');print(json.dumps(receipt))
