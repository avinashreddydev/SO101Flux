"""Robot-side client: drives a physical SO-101 by calling a remote FLUX.3 inference server
over HTTP, once per control tick.

Run this on the machine physically connected to the SO-101 (USB port + two cameras). The GPU
server (this repo's ``flux3_lerobot_server.py``) can run on a different machine on the same
network; only images/state/JSON cross the wire, nothing from `lerobot` needs installing there.

This machine only needs a lightweight LeRobot install (robot drivers + cameras), NOT the
`flux3` extra, torch, or a GPU:

    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e .
    uv pip install requests pillow

Calibrate the arm first if you haven't (see LeRobot's SO-101 setup docs):

    lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101

Then run this script, e.g.:

    python robot_client.py \\
      --server http://SERVER_IP:8000 \\
      --api-key "$FLUX_API_KEY" \\
      --port /dev/ttyACM0 --robot-id so101 \\
      --scene-camera 0 --wrist-camera 1 \\
      --task "put the blue box into the container"

Safety is entirely the client's job: the server only returns numbers. This script enforces a
per-tick joint-motion cap (via LeRobot's `max_relative_target`), a response staleness timeout,
and stops the robot (holds position, disables torque) on any error or Ctrl-C.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

import requests
from PIL import Image

# Joint order the SO-101 checkpoint expects: shoulder pan, shoulder lift, elbow flex,
# wrist flex, wrist roll, gripper. Arm joints in degrees, gripper in percentage points.
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Per-tick safety cap on how far any single joint may move (LeRobot's max_relative_target).
# Degrees for the five arm joints, percentage points for the gripper. Tune down for a first run.
DEFAULT_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 8.0,
    "shoulder_lift": 8.0,
    "elbow_flex": 8.0,
    "wrist_flex": 8.0,
    "wrist_roll": 8.0,
    "gripper": 15.0,
}

# Drop a response and stop locally if a /step call takes longer than this (server hiccup,
# network stall, or a dropped connection) rather than apply a stale command.
STEP_TIMEOUT_S = 5.0


def frame_to_jpeg_bytes(frame) -> bytes:
    """OpenCV/np frame (H, W, 3) RGB -> JPEG bytes for the multipart upload."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ServerClient:
    def __init__(self, base_url: str, api_key: str, session_id: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.session_id = session_id

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/v1/health", timeout=STEP_TIMEOUT_S)
        r.raise_for_status()
        return r.json()

    def load_model(self, checkpoint: str) -> dict:
        r = requests.post(
            f"{self.base_url}/v1/model/load",
            headers={**self.headers, "Content-Type": "application/json"},
            json={"checkpoint": checkpoint},
            timeout=600,  # first load downloads/moves a ~20 GB model
        )
        r.raise_for_status()
        return r.json()

    def reset(self) -> None:
        r = requests.post(
            f"{self.base_url}/v1/sessions/{self.session_id}/reset",
            headers=self.headers,
            timeout=STEP_TIMEOUT_S,
        )
        r.raise_for_status()

    def step(self, scene_jpeg: bytes, wrist_jpeg: bytes, state: list[float], task: str) -> list[float]:
        r = requests.post(
            f"{self.base_url}/v1/sessions/{self.session_id}/step",
            headers=self.headers,
            files={
                "scene": ("scene.jpg", scene_jpeg, "image/jpeg"),
                "wrist": ("wrist.jpg", wrist_jpeg, "image/jpeg"),
            },
            data={"state": json.dumps(state), "task": task},
            timeout=STEP_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()["action"]


def build_robot(port: str, robot_id: str, scene_camera, wrist_camera, fps: int, max_relative_target: dict):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        use_degrees=True,
        max_relative_target=max_relative_target,
        cameras={
            "scene": OpenCVCameraConfig(index_or_path=scene_camera, width=640, height=480, fps=fps),
            "wrist": OpenCVCameraConfig(index_or_path=wrist_camera, width=640, height=480, fps=fps),
        },
    )
    return SO101Follower(config)


def run(args: argparse.Namespace) -> None:
    client = ServerClient(args.server, args.api_key, args.session_id)

    print("Checking server health...")
    health = client.health()
    print(health)
    if not health["cuda_available"]:
        print("WARNING: server reports no CUDA device available.", file=sys.stderr)

    if not health["model_loaded"] or health.get("checkpoint") != args.checkpoint:
        print(f"Loading {args.checkpoint} on the server (can take a few minutes on first run)...")
        print(client.load_model(args.checkpoint))

    max_relative_target = dict(DEFAULT_MAX_RELATIVE_TARGET)
    robot = build_robot(
        args.port, args.robot_id, args.scene_camera, args.wrist_camera, args.fps, max_relative_target
    )
    robot.connect()
    print("Robot connected. Resetting session and starting control loop.")
    print("Press Ctrl+C to stop.")

    client.reset()
    period = 1.0 / args.fps
    tick = 0
    try:
        while args.duration is None or tick * period < args.duration:
            tick_start = time.perf_counter()

            obs = robot.get_observation()
            state = [float(obs[f"{joint}.pos"]) for joint in JOINT_ORDER]
            scene_jpeg = frame_to_jpeg_bytes(obs["scene"])
            wrist_jpeg = frame_to_jpeg_bytes(obs["wrist"])

            try:
                action_values = client.step(scene_jpeg, wrist_jpeg, state, args.task)
            except requests.RequestException as exc:
                print(f"Server call failed ({exc}); stopping.", file=sys.stderr)
                break

            action = {f"{joint}.pos": value for joint, value in zip(JOINT_ORDER, action_values, strict=True)}
            robot.send_action(action)

            tick += 1
            elapsed = time.perf_counter() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
            elif elapsed > 2 * period:
                print(f"WARNING: tick {tick} took {elapsed * 1000:.0f} ms (target {period * 1000:.0f} ms)")
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="Base URL of the inference server, e.g. http://192.168.1.50:8000")
    parser.add_argument("--api-key", required=True, help="Value of the server's FLUX_API_KEY")
    parser.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    parser.add_argument("--session-id", default="so101-episode-1")
    parser.add_argument("--port", required=True, help="Serial port of the SO-101, e.g. /dev/ttyACM0")
    parser.add_argument("--robot-id", required=True, help="Calibration ID used with lerobot-calibrate")
    parser.add_argument("--scene-camera", default=0, help="OpenCV camera index or path for the scene view")
    parser.add_argument("--wrist-camera", default=1, help="OpenCV camera index or path for the wrist view")
    parser.add_argument("--fps", type=int, default=30, help="Control loop rate; must match the checkpoint's fps")
    parser.add_argument("--task", required=True, help="Task instruction string, e.g. 'put the blue box into the container'")
    parser.add_argument("--duration", type=float, default=None, help="Episode length in seconds (default: run until Ctrl+C)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
