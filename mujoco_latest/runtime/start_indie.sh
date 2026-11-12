#!/usr/bin/env bash
# Independent-port streaming launch (does NOT touch default 49100/47998 used by handstand replay).
set -eu
cd /home/ubuntu/Go2W/interactive_stage1_20261104
export OMNI_KIT_ACCEPT_EULA=YES
exec /home/ubuntu/envs/isaac62/bin/python -u main.py --livestream 1 --device cuda:0 \
  --bridge_port 8766 --kit_args='--/exts/omni.kit.livestream.app/primaryStream/publicIp=106.75.23.13 --/exts/omni.kit.livestream.app/primaryStream/signalPort=49101 --/exts/omni.kit.livestream.app/primaryStream/streamPort=48000' \
  "$@"
