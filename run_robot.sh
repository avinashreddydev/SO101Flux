#!/usr/bin/env bash
# Drive the physical SO-101 by calling a running FLUX.3 Action inference server (run_model.sh)
# once per control tick. Run this on the machine physically connected to the SO-101 (USB) and
# its two cameras — it can be this same machine or a separate one on the same network as the
# server; either way it reuses this repo's .venv for LeRobot's robot/camera drivers (installing
# the Feetech motor SDK into it on first run).
#
# Usage:
#   FLUX_SERVER=http://SERVER_IP:8000 \
#     ./run_robot.sh --port /dev/ttyACM0 --robot-id so101 \
#                     --scene-camera 0 --wrist-camera 1 \
#                     --task "put the blue box into the container" [--duration 30]
#
# All flags after the script name are forwarded to remote_inference/robot_client.py
# (run it with --help for the full list). FLUX_SERVER defaults to this GPU server's LAN
# address; override it if the server moves.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
SERVER="${FLUX_SERVER:-http://10.171.42.11:8000}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "error: $VENV not found. On a dedicated robot machine (no GPU needed), set up a" >&2
  echo "lightweight venv instead — see remote_inference/ROBOT_CLIENT_USAGE.md." >&2
  exit 1
fi

# feetech-servo-sdk (SO-101's motor bus) isn't installed by the flux3 extra used for the
# model server; add it on first run. No torch/GPU deps pulled in by this extra.
if ! "$VENV/bin/python" -c "import scservo_sdk" >/dev/null 2>&1; then
  echo "Installing SO-101 hardware drivers (feetech-servo-sdk, pyserial) into $VENV ..."
  uv pip install --python "$VENV/bin/python" -e "./lerobot[feetech]"
fi
"$VENV/bin/python" -c "import requests" >/dev/null 2>&1 || uv pip install --python "$VENV/bin/python" requests

echo "Server: $SERVER"
exec "$VENV/bin/python" remote_inference/robot_client.py \
  --server "$SERVER" \
  "$@"
