"""FastAPI inference service for the FLUX.3 Action SO-101 policy, backed by LeRobot's
``lerobot.policies.flux3`` integration.

Same HTTP contract as ``flux3_fastapi_server.py`` (health / model.load / sessions.reset /
sessions.step), so robot-side clients do not need to change. Swapped to this backend because
the standalone ``flux-action`` package's two-pass CFG sampling exceeds 24 GiB of GPU memory
during inference on this machine, even with the text encoder offloaded to CPU. This backend
instead splits the policy across two GPUs: the DiT + video VAE on ``cuda:0`` and the frozen
Qwen3-VL text encoder on ``cuda:1``, which fits and has been verified to run a full rollout.

This process only performs model inference. It never connects to robot hardware. The robot-side
client must enforce joint limits, stale-response timeouts, and an emergency stop before applying
any returned command.

The policy is stateful. Call ``/step`` once per control tick, including ticks served from the
policy's internal action queue, and reset the session at every episode boundary or intervention.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from typing import Annotated

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

DEFAULT_CHECKPOINT = "black-forest-labs/flux-3-action-so101"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096


class LoadModelRequest(BaseModel):
    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str | None = None
    # Device for the DiT + video VAE. The text encoder always goes on a second CUDA device
    # when one is available, otherwise it falls back to CPU (matches LeRobot's documented
    # low-VRAM path, just slower to encode an uncached instruction).
    device: str = "cuda:0"
    text_encoder_device: str | None = None


class LoadModelResponse(BaseModel):
    loaded: bool
    checkpoint: str
    revision: str | None
    device: str
    text_encoder_device: str
    camera_keys: list[str]
    fps: float
    action_steps: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str | None
    device: str | None
    active_session: str | None
    cuda_available: bool
    cuda_device_count: int


class StepResponse(BaseModel):
    session_id: str
    action: list[float]
    latency_ms: float
    units: str = "arm joints: degrees; gripper: percentage points"


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Require X-API-Key when FLUX_API_KEY is configured on the server."""
    expected = os.getenv("FLUX_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def decode_rgb_image(payload: bytes, field_name: str) -> torch.Tensor:
    if not payload:
        raise HTTPException(status_code=422, detail=f"{field_name} image is empty")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field_name} image exceeds 10 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
            if max(image.size) > MAX_IMAGE_SIDE:
                raise HTTPException(status_code=422, detail=f"{field_name} image is too large")
            array = np.asarray(image, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid {field_name} image") from exc
    # (H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]; the preprocessor resizes to the
    # checkpoint's canvas and handles normalization from there.
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def parse_state(raw: str) -> list[float]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="state must be a JSON array") from exc
    if not isinstance(values, list) or len(values) != 6:
        raise HTTPException(status_code=422, detail="state must contain exactly six values")
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="state values must be numeric") from exc
    if not all(np.isfinite(result)):
        raise HTTPException(status_code=422, detail="state values must be finite")
    return result


class Flux3SO101Engine:
    """Own one model, its processors, and one active SO-101 control session."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.checkpoint: str | None = None
        self.revision: str | None = None
        self.device: torch.device | None = None
        self.text_encoder_device: torch.device | None = None
        self.active_session: str | None = None

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=self.policy is not None,
            checkpoint=self.checkpoint,
            device=str(self.device) if self.device else None,
            active_session=self.active_session,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )

    def load(self, request: LoadModelRequest) -> LoadModelResponse:
        with self.lock:
            device = torch.device(request.device)
            if device.type != "cuda":
                raise HTTPException(status_code=422, detail="FLUX.3 SO-101 inference requires CUDA")
            if not torch.cuda.is_available():
                raise HTTPException(status_code=503, detail="CUDA is not available on this server")

            if request.text_encoder_device:
                text_device = torch.device(request.text_encoder_device)
            elif torch.cuda.device_count() > 1:
                other = 1 - device.index if device.index is not None else 1
                text_device = torch.device(f"cuda:{other}")
            else:
                text_device = torch.device("cpu")

            try:
                from lerobot.policies.factory import make_pre_post_processors
                from lerobot.policies.flux3 import Flux3Policy
                from lerobot.policies.flux3.configuration_flux3 import Flux3Config
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="lerobot[flux3] is not installed in the server environment",
                ) from exc

            try:
                config = Flux3Config.from_pretrained(request.checkpoint, revision=request.revision)
                config.device = "cpu"  # load on CPU first; avoids a single-GPU OOM during .to()
                policy = Flux3Policy.from_pretrained(request.checkpoint, revision=request.revision, config=config)
                policy.eval()
                policy.dit.to(device)
                policy.frozen.video_vae.module.to(device)
                policy.frozen.text_encoder.to(text_device)
                policy.config.device = str(device)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy.config, pretrained_path=request.checkpoint
                )
                policy.reset()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"model load failed: {exc}") from exc

            previous = self.policy
            self.policy = policy
            self.preprocessor = preprocessor
            self.postprocessor = postprocessor
            self.checkpoint = request.checkpoint
            self.revision = request.revision
            self.device = device
            self.text_encoder_device = text_device
            self.active_session = None
            if previous is not None:
                del previous
                torch.cuda.empty_cache()

            return LoadModelResponse(
                loaded=True,
                checkpoint=request.checkpoint,
                revision=request.revision,
                device=str(device),
                text_encoder_device=str(text_device),
                camera_keys=list(policy.config.camera_order),
                fps=float(policy.config.fps),
                action_steps=int(policy.config.n_action_steps),
            )

    def reset(self, session_id: str) -> None:
        with self.lock:
            if self.policy is None:
                raise HTTPException(status_code=409, detail="load the model first")
            self.policy.reset()
            self.active_session = session_id

    def step(
        self,
        session_id: str,
        scene: torch.Tensor,
        wrist: torch.Tensor,
        state: list[float],
        task: str,
    ) -> StepResponse:
        with self.lock:
            if self.policy is None or self.device is None:
                raise HTTPException(status_code=409, detail="load the model first")
            if self.active_session != session_id:
                raise HTTPException(
                    status_code=409,
                    detail="reset this session before sending observations",
                )
            if not task.strip():
                raise HTTPException(status_code=422, detail="task must not be empty")

            obs = {
                "observation.images.scene": scene.to(self.device, non_blocking=True),
                "observation.images.wrist": wrist.to(self.device, non_blocking=True),
                "observation.state": torch.tensor([state], dtype=torch.float32, device=self.device),
                "task": task.strip(),
            }
            started = time.perf_counter()
            try:
                prepared = self.preprocessor(obs)
                action = self.policy.select_action(prepared)
                command = self.postprocessor(action)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            except Exception as exc:
                # Do not continue from partially updated history after an inference error.
                self.policy.reset()
                self.active_session = None
                raise HTTPException(status_code=500, detail=f"inference failed; session reset: {exc}") from exc

            latency_ms = (time.perf_counter() - started) * 1000
            values = command[0].detach().float().cpu().tolist()
            if len(values) != 6 or not all(np.isfinite(values)):
                self.policy.reset()
                self.active_session = None
                raise HTTPException(status_code=500, detail="model returned an invalid action; session reset")
            return StepResponse(session_id=session_id, action=values, latency_ms=latency_ms)


engine = Flux3SO101Engine()
app = FastAPI(
    title="FLUX.3 Action SO-101 inference server (LeRobot backend)",
    version="1.0.0",
    description="Stateful model inference only; robot control and safety remain on the client.",
)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return engine.health()


@app.post(
    "/v1/model/load",
    response_model=LoadModelResponse,
    dependencies=[Depends(require_api_key)],
)
def load_model(request: LoadModelRequest) -> LoadModelResponse:
    return engine.load(request)


@app.post("/v1/sessions/{session_id}/reset", dependencies=[Depends(require_api_key)])
def reset_session(session_id: str) -> dict[str, str]:
    if not session_id.strip() or len(session_id) > 128:
        raise HTTPException(status_code=422, detail="invalid session ID")
    engine.reset(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post(
    "/v1/sessions/{session_id}/step",
    response_model=StepResponse,
    dependencies=[Depends(require_api_key)],
)
async def step(
    session_id: str,
    scene: Annotated[UploadFile, File(description="Scene camera JPEG/PNG")],
    wrist: Annotated[UploadFile, File(description="Wrist camera JPEG/PNG")],
    state: Annotated[str, Form(description="Six raw joint values as a JSON array")],
    task: Annotated[str, Form(description="Task instruction")],
) -> StepResponse:
    # Reading uploads is asynchronous; GPU inference is serialized by the engine lock.
    scene_bytes = await scene.read(MAX_IMAGE_BYTES + 1)
    wrist_bytes = await wrist.read(MAX_IMAGE_BYTES + 1)
    scene_tensor = decode_rgb_image(scene_bytes, "scene")
    wrist_tensor = decode_rgb_image(wrist_bytes, "wrist")
    state_values = parse_state(state)
    return engine.step(session_id, scene_tensor, wrist_tensor, state_values, task)
