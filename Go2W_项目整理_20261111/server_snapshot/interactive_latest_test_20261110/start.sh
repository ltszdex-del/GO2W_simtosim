#!/usr/bin/env bash
set -eu
cd /home/ubuntu/Go2W/interactive_latest_test_20261110
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH=/home/ubuntu/Go2W/stream_noise_fix_20261104/ngx_lib:${LD_LIBRARY_PATH:-}
exec /home/ubuntu/envs/isaac62/bin/python -u main.py --livestream 1 --device cuda:0 --kit_args='--/exts/omni.kit.livestream.app/primaryStream/publicIp=106.75.23.13 --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998 --/ngx/enabled=true' "$@"
