# Running FLUX.3 Action SO-101 as a remote inference server

Two machines are involved:

- **GPU server** (this machine, `so101flux/`): loads the 7B model and answers `/step` requests.
  Already running here as `flux3_lerobot_server.py`, backed by the `lerobot` install in
  `so101flux/.venv`.
- **Robot machine**: physically connected to the SO-101 (USB) and its two cameras. Runs
  `robot_client.py` in a loop, sending observations to the server and applying the actions it
  gets back. It does **not** need a GPU, torch, or the `flux3` extra — just LeRobot's robot
  drivers.

Only JPEGs + small JSON payloads cross the network each tick; nothing from `lerobot`'s ML stack
needs to run on the robot machine.

## 1. GPU server (already running here)

```bash
cd /home/av131693/Desktop/research/so101flux
./run_model.sh
```

This starts the server, waits for it to come up, and loads the checkpoint automatically,
printing the LAN URL to use from the robot machine when it's ready. This machine has two 24 GiB
GPUs; the server puts the DiT + video VAE on `cuda:0` (~16 GiB) and the Qwen3-VL text encoder on
`cuda:1` (~9 GiB) — verified working end-to-end. A single 24 GiB GPU is **not** enough for this
checkpoint's inference (only loading fits; the diffusion sampling step itself needs more
headroom), so keep both GPUs free for this process. Press Ctrl+C to stop it.

**Security (dev setup, no auth)**: `run_model.sh` binds `0.0.0.0` and starts the server with no
API key — fine on a trusted LAN for development, not for anything public-facing. The server
(`flux3_lerobot_server.py`) does support an `X-API-Key` check when `FLUX_API_KEY` is set in its
environment, and `robot_client.py --api-key` can send one; `run_model.sh` just doesn't wire that
up by default. Add it back in both scripts if you need it.

## 2. Robot machine setup

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .           # robot/camera drivers only — no torch/flux3 needed here
uv pip install requests pillow
```

Copy `robot_client.py` from this repo's `remote_inference/` folder onto the robot machine (scp,
USB drive, git — whatever's convenient); it's a single self-contained script.

Calibrate the arm if you haven't already:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101
```

Confirm your two camera device indices/paths (`ls /dev/video*` or LeRobot's camera-finder
utility) and which one is the fixed **scene** view vs. the **wrist**-mounted view — the
checkpoint was trained with `scene` on the left, `wrist` on the right of its internal canvas;
swapping them silently degrades the policy rather than erroring.

## 3. Run a control loop

```bash
python robot_client.py \
  --server http://SERVER_IP:8000 \
  --api-key "$FLUX_API_KEY" \
  --port /dev/ttyACM0 --robot-id so101 \
  --scene-camera 0 --wrist-camera 1 \
  --task "put the blue box into the container" \
  --duration 30
```

Replace `SERVER_IP` with the GPU machine's LAN address and `FLUX_API_KEY` with the value from
`remote_inference/.api_key` on the server. Omit `--duration` to run until Ctrl+C.

What it does each tick, at `--fps` (default 30, matching the checkpoint):

1. Reads the arm's joint positions and both camera frames.
2. POSTs them to `/v1/sessions/<id>/step` and gets back one absolute 6-value command.
3. Sends that command to the arm, capped by `max_relative_target` (a few degrees per tick,
   defined near the top of `robot_client.py`) so a bad or delayed response can't slam the arm.
4. Sleeps out the rest of the tick period.

The model replans every 32 ticks (~1 second at 30 Hz); those ticks take ~2-3 seconds of GPU
compute and this script's `STEP_TIMEOUT_S` (5s) accounts for that. Ticks served from the
policy's internal queue return in a few milliseconds.

**Reset before every new episode** and after any manual intervention — the script's `reset()`
call at startup covers the first episode; for a multi-episode session, call
`client.reset()` again (or restart the script) between episodes so the policy's observation
history doesn't carry over stale context.

## Safety notes (client-side, already in `robot_client.py`)

- **Per-tick motion cap**: `DEFAULT_MAX_RELATIVE_TARGET` in `robot_client.py` limits how far any
  single joint can move per tick, regardless of what the model returns. Start conservative
  (the defaults are deliberately small) and only raise it once you trust the setup.
- **Stale/slow response handling**: a `/step` call that exceeds `STEP_TIMEOUT_S` raises and stops
  the loop rather than applying a delayed command.
- **Torque/E-stop**: keep a physical means to cut power to the arm within reach. This script
  disconnects (which disables torque, per `disable_torque_on_disconnect`) on any error or
  Ctrl+C, but that's not a substitute for a hardware stop.
- Nothing here validates that `--task` matches what's actually in front of the camera, or that
  the scene is safe (no people/obstacles in the arm's workspace) — that's on you before each run.
