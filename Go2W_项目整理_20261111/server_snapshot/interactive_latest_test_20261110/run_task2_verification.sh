set -eu
cd /home/ubuntu/Go2W/interactive_stage1_20261104
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
/home/ubuntu/envs/isaac62/bin/python -m unittest -v test_control test_keyboard_router test_stream_ui test_task2
folder="logs/task2_verify_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$folder"
echo "$folder" > logs/task2_latest.txt
OMNI_KIT_ACCEPT_EULA=YES timeout 240 /home/ubuntu/envs/isaac62/bin/python -u verify_task2_runtime.py --livestream 0 --viz none --device cuda:0 --log_dir "$folder" > "$folder/console.log" 2>&1
test -s "$folder/report.json"
cat "$folder/report.json"
