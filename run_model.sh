#!/usr/bin/env bash
# Start the FLUX.3 Action SO-101 inference server on this (GPU) machine and load the model.
# Run this first; run_robot.sh (on the robot-connected machine) talks to it over HTTP.
#
# Usage:
#   ./run_model.sh                          # defaults: 0.0.0.0:8000, black-forest-labs/flux-3-action-so101
#   FLUX_PORT=8001 ./run_model.sh
#   FLUX_CHECKPOINT=black-forest-labs/flux-3-action-droid ./run_model.sh
#
# Requires both GPUs free: the DiT + video VAE go on cuda:0 (~16 GiB), the Qwen3-VL text
# encoder on cuda:1 (~9 GiB) — see remote_inference/ROBOT_CLIENT_USAGE.md for why a single
# 24 GiB GPU isn't enough for this checkpoint's inference.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
HOST="${FLUX_HOST:-0.0.0.0}"
PORT="${FLUX_PORT:-8000}"
CHECKPOINT="${FLUX_CHECKPOINT:-black-forest-labs/flux-3-action-so101}"

if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "error: $VENV is missing lerobot/uvicorn. Set it up first:" >&2
  echo "  git clone https://github.com/huggingface/lerobot.git" >&2
  echo "  uv venv --python 3.13 $VENV" >&2
  echo "  uv pip install --python $VENV/bin/python -e ./lerobot'[flux3]'" >&2
  echo "  uv pip install --python $VENV/bin/python natten==0.21.6+torch2110cu128 --find-links https://whl.natten.org" >&2
  echo "  uv pip install --python $VENV/bin/python -r remote_inference/server-requirements.txt" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || [ "$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)" -lt 2 ]; then
  echo "warning: expected 2 GPUs (DiT+VAE on cuda:0, text encoder on cuda:1); proceeding anyway." >&2
fi

echo "Starting server on $HOST:$PORT (no API key — dev mode)..."
"$VENV/bin/uvicorn" flux3_lerobot_server:app \
  --app-dir remote_inference \
  --host "$HOST" --port "$PORT" --workers 1 &
SERVER_PID=$!

cleanup() {
  echo
  echo "Stopping server (pid $SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for the server to come up..."
until curl -sf "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: server process exited before becoming healthy; check the output above." >&2
    exit 1
  fi
  sleep 1
done

echo "Loading $CHECKPOINT (first load can take a few minutes)..."
curl -sf -X POST "http://127.0.0.1:$PORT/v1/model/load" \
  -H 'Content-Type: application/json' \
  -d "{\"checkpoint\": \"$CHECKPOINT\"}"
echo
echo

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF
Server ready.
  URL (LAN):    http://${LAN_IP:-<this-machine-ip>}:$PORT
  URL (local):  http://127.0.0.1:$PORT

On the robot machine, run run_robot.sh with:
  FLUX_SERVER=http://${LAN_IP:-<this-machine-ip>}:$PORT ./run_robot.sh \\
    --port /dev/ttyACM0 --robot-id so101 --scene-camera 0 --wrist-camera 1 --task "..."

Press Ctrl+C to stop the server.
EOF

wait "$SERVER_PID"
