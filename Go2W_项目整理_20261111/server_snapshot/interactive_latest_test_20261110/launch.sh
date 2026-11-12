set -eu
cd /home/ubuntu/Go2W/interactive_stage1_20261104
/home/ubuntu/envs/isaac62/bin/python -m unittest -v test_control test_keyboard_router test_stream_ui test_task2
python3 - <<'PY'
import socket
for kind,port in [(socket.SOCK_STREAM,49100),(socket.SOCK_DGRAM,47998),(socket.SOCK_STREAM,8765)]:
    s=socket.socket(socket.AF_INET,kind)
    if kind==socket.SOCK_STREAM:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try:s.bind(('0.0.0.0',port))
    except OSError as e:raise SystemExit(f'Port {port} occupied: abort without stopping any process: {e}')
    finally:s.close()
PY
folder="logs/acceptance_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$folder"
nohup bash start.sh --steps 0 --log_dir "$folder" > "$folder/console.log" 2>&1 < /dev/null &
echo $! > "$folder/pid"
echo "$folder" > logs/current.txt
echo "PID=$(cat "$folder/pid") LOG=$folder"
