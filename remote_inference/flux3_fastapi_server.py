"""FastAPI inference service for the FLUX.3 Action SO-101 policy.

This process only performs model inference. It never connects to robot hardware.
The robot-side client must enforce joint limits, stale-response timeouts, and an
emergency stop before applying any returned command.

The released SO-101 policy is stateful. Call ``/step`` once per control tick,
including ticks served from the policy's internal action queue, and reset the
session at every episode boundary or intervention.
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
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError


DEFAULT_CHECKPOINT = "black-forest-labs/flux-3-action-so101"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096


class LoadModelRequest(BaseModel):
    checkpoint: str = DEFAULT_CHECKPOINT
    revision: str | None = None
    device: str = "cuda"
    offload_text_encoder: bool = False


class LoadModelResponse(BaseModel):
    loaded: bool
    checkpoint: str
    revision: str | None
    device: str
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
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


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
    """Own one model and one active SO-101 control session."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.policy = None
        self.checkpoint: str | None = None
        self.revision: str | None = None
        self.device: torch.device | None = None
        self.active_session: str | None = None

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=self.policy is not None,
            checkpoint=self.checkpoint,
            device=str(self.device) if self.device else None,
            active_session=self.active_session,
            cuda_available=torch.cuda.is_available(),
        )

    def load(self, request: LoadModelRequest) -> LoadModelResponse:
        with self.lock:
            device = torch.device(request.device)
            if device.type != "cuda":
                raise HTTPException(status_code=422, detail="FLUX.3 SO-101 inference requires CUDA")
            if not torch.cuda.is_available():
                raise HTTPException(status_code=503, detail="CUDA is not available on this server")

            try:
                from flux_action.inference.so101 import load_policy
            except ImportError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="flux-action is not installed in the server environment",
                ) from exc

            try:
                policy = load_policy(request.checkpoint, revision=request.revision)
                # Offload must be set before .to(device): _apply() only skips moving the text
                # encoder's weights to the GPU when the flag is already set, so setting it
                # afterward doesn't avoid the transient peak memory of moving everything at once.
                if request.offload_text_encoder:
                    policy.set_text_encoder_offload(True)
                policy = policy.to(device).eval()
                policy.reset()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"model load failed: {exc}") from exc

            previous = self.policy
            self.policy = policy
            self.checkpoint = request.checkpoint
            self.revision = request.revision
            self.device = device
            self.active_session = None
            if previous is not None:
                del previous
                torch.cuda.empty_cache()

            return LoadModelResponse(
                loaded=True,
                checkpoint=request.checkpoint,
                revision=request.revision,
                device=str(device),
                camera_keys=list(policy.config.camera_keys),
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
            if scene.shape != wrist.shape:
                raise HTTPException(
                    status_code=422,
                    detail="scene and wrist images must have the same dimensions",
                )
            if not task.strip():
                raise HTTPException(status_code=422, detail="task must not be empty")

            batch = {
                "images.scene": scene.to(self.device, non_blocking=True),
                "images.wrist": wrist.to(self.device, non_blocking=True),
                "state": torch.tensor([state], dtype=torch.float32, device=self.device),
                "task": [task.strip()],
            }
            started = time.perf_counter()
            try:
                action = self.policy.select_action(batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            except Exception as exc:
                # Do not continue from partially updated history after an inference error.
                self.policy.reset()
                self.active_session = None
                raise HTTPException(status_code=500, detail=f"inference failed; session reset: {exc}") from exc

            latency_ms = (time.perf_counter() - started) * 1000
            values = action[0].detach().float().cpu().tolist()
            if len(values) != 6 or not all(np.isfinite(values)):
                self.policy.reset()
                self.active_session = None
                raise HTTPException(status_code=500, detail="model returned an invalid action; session reset")
            return StepResponse(session_id=session_id, action=values, latency_ms=latency_ms)


engine = Flux3SO101Engine()
app = FastAPI(
    title="FLUX.3 Action SO-101 inference server",
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

