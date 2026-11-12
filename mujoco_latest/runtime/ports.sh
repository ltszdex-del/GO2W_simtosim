date -u
ps -eo pid,etimes,args | grep -E 'main.py|scripts/play.py|scripts/train.py|livestream' | grep -v grep | grep -v telemetry || true
python3 - <<'PY'
import pathlib,socket
inodes=set()
for proto in ('tcp','tcp6','udp','udp6'):
    for row in pathlib.Path('/proc/net/'+proto).read_text().splitlines()[1:]:
        fields=row.split();port=int(fields[1].split(':')[1],16)
        if port in (49100,47998,8765,49101,48000,8766):
            print(proto,'port',port,'state',fields[3],'inode',fields[9]);inodes.add(fields[9])
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        for fd in (proc/'fd').iterdir():
            try:target=fd.readlink().as_posix()
            except OSError:continue
            if target.startswith('socket:[') and target[8:-1] in inodes:
                print('OWNER',proc.name,(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'));break
    except (OSError,PermissionError):pass
for kind,port in [(socket.SOCK_STREAM,49100),(socket.SOCK_DGRAM,47998),(socket.SOCK_STREAM,8765)]:
    s=socket.socket(socket.AF_INET,kind)
    if kind==socket.SOCK_STREAM:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try:s.bind(('0.0.0.0',port));print('FREE',port)
    except OSError as e:print('BUSY',port,str(e))
    finally:s.close()
PY
