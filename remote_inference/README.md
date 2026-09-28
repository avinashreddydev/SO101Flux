# FLUX.3 Action SO-101 inference server

This service loads the official SO-101 policy and returns one absolute robot
command for each control tick. It does not connect to the robot or execute
actions.

## Install on the GPU server

Follow the official `flux-action` setup first:

```bash
git clone https://github.com/black-forest-labs/flux-action.git
cd flux-action
uv sync --locked --extra encoders
uv pip install --python .venv/bin/python \
  'natten==0.21.6+torch2100cu128' \
  -f https://whl.natten.org/
```

The NATTEN suffix must match the server's Torch and CUDA versions. Then install
the HTTP dependencies into that environment:

```bash
uv pip install --python .venv/bin/python \
  -r /path/to/SO_ARM/remote_inference/server-requirements.txt
```

Authenticate on the server if the model repository requires access:

```bash
uv run hf auth login
```

## Start the API

Set a strong API key and start exactly one worker. Multiple workers would load
multiple model copies and maintain conflicting policy histories.

```bash
export FLUX_API_KEY='replace-with-a-long-random-secret'

uv run uvicorn flux3_fastapi_server:app \
  --app-dir /path/to/SO_ARM/remote_inference \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Keep port 8000 on a trusted LAN or behind a VPN/SSH tunnel. Do not expose this
development server directly to the public internet.

## Load the model

```bash
curl -X POST http://SERVER_IP:8000/v1/model/load \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $FLUX_API_KEY" \
  -d '{
    "checkpoint": "black-forest-labs/flux-3-action-so101",
    "device": "cuda",
    "offload_text_encoder": false
  }'
```

Loading can take several minutes the first time. Check status with:

```bash
curl http://SERVER_IP:8000/v1/health
```

## Start or reset an episode

Reset before every new episode and after any human intervention:

```bash
curl -X POST http://SERVER_IP:8000/v1/sessions/so101-episode-1/reset \
  -H "X-API-Key: $FLUX_API_KEY"
```

## Send one control tick

The released checkpoint expects two RGB views with the same dimensions. State
order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, and
gripper. Arm joints are degrees and the gripper is percentage points.

```bash
curl -X POST http://SERVER_IP:8000/v1/sessions/so101-episode-1/step \
  -H "X-API-Key: $FLUX_API_KEY" \
  -F scene=@scene.jpg \
  -F wrist=@wrist.jpg \
  -F 'state=[0, 0, 0, 0, 0, 50]' \
  -F 'task=place the box in the container'
```

Call `/step` on every 30 Hz control tick, including ticks where inference is
served from the model's internal queue. The response contains one six-value
absolute command. The robot client must reject stale, out-of-range, malformed,
or late responses and stop locally when connectivity is lost.

