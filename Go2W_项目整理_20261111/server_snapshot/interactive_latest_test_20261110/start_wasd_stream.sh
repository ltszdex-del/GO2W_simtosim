#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
python3 - <<'PY'
import socket
for kind,port in [(socket.SOCK_STREAM,49100),(socket.SOCK_DGRAM,47998),(socket.SOCK_STREAM,8765)]:
    with socket.socket(socket.AF_INET,kind) as s:
        if kind==socket.SOCK_STREAM:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('0.0.0.0',port))
PY
mkdir -p logs
log="logs/wasd_live_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$log"
nohup bash start.sh --log_dir "$log" "$@" > "$log/console.log" 2>&1 < /dev/null &
echo $! > "$log/pid"
echo "$log" > logs/current.txt
echo "PID=$! LOG=$PWD/$log"
