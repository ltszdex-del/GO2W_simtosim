from pathlib import Path
import urllib.request,json,time,os
r=Path('/home/ubuntu/Go2W/interactive_latest_test_20261110')
folder=Path((r/'logs/current.txt').read_text().strip());receipt=json.loads((folder/'launch.json').read_text())
os.kill(receipt['pid'],0)
def get():
    with urllib.request.urlopen('http://127.0.0.1:8765/status',timeout=5) as response:return json.load(response)
a=get();time.sleep(2);b=get()
assert b['physics']['step']>a['physics']['step'],b
assert b['policy_errors']=={} and b['mode']!='error',b
assert receipt['policies']=={k:v['checkpoint'] for k,v in b['policies'].items()}
ports=[]
for proto in ('tcp','tcp6','udp','udp6'):
    for line in Path('/proc/net/'+proto).read_text().splitlines()[1:]:
        f=line.split();port=int(f[1].split(':')[1],16)
        if port in (49100,47998,8765):ports.append(dict(protocol=proto,port=port,state=f[3]))
assert any(p['port']==49100 and p['state']=='0A' for p in ports),ports
assert (folder/'preview.png').exists(),'Awaiting screenshot'
result=dict(pid=receipt['pid'],folder=str(folder),mode=b['mode'],step=b['physics']['step'],
    height=b['physics']['xyz'][2],policy_errors=b['policy_errors'],simulation=b.get('simulation'),
    ports=ports,preview=str(folder/'preview.png'))
(folder/'verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
