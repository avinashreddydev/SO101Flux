import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.flux3.configuration_flux3 import Flux3Config
from lerobot.utils.constants import OBS_STATE

repo_id = "black-forest-labs/flux-3-action-so101"

# The DiT (~7B) + Qwen3-VL text encoder (~4B) + video VAE together approach ~22 GiB in bf16,
# which doesn't fit loading-time overhead on a single 24 GiB GPU. Load on CPU first, then
# split components across the two GPUs.
print("Loading config...")
config = Flux3Config.from_pretrained(repo_id)
config.device = "cpu"

print("Loading policy weights on CPU...")
policy = Flux3Policy.from_pretrained(repo_id, config=config)
policy.eval()

print("Placing DiT + video VAE on cuda:0, text encoder on cuda:1...")
policy.dit.to("cuda:0")
policy.frozen.video_vae.module.to("cuda:0")
policy.frozen.text_encoder.to("cuda:1")
policy.config.device = "cuda:0"

print("Building pre/post processors...")
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=repo_id)

cfg = policy.config
print("camera_order:", cfg.camera_order)
print("action_dim:", cfg.action_dim, "n_obs_steps:", cfg.n_obs_steps)
print("chunk_size:", cfg.chunk_size, "n_action_steps:", cfg.n_action_steps)
for cam in cfg.camera_order:
    print(cam, "shape:", cfg.input_features[cam].shape)

device = "cuda:0"
policy.reset()

print("\nRunning a synthetic rollout smoke test (random frames, fixed task string)...")
n_ticks = cfg.n_obs_steps + 2
for tick in range(n_ticks):
    obs = {"task": "put the blue box into the container"}
    for cam in cfg.camera_order:
        c, h, w = cfg.input_features[cam].shape
        obs[cam] = torch.rand(1, c, h, w, device=device)
    obs[OBS_STATE] = torch.rand(1, cfg.action_dim, device=device)

    prepared = preprocessor(obs)
    action = policy.select_action(prepared)
    command = postprocessor(action)
    print(f"tick {tick}: action shape {tuple(action.shape)}, command shape {tuple(command.shape)}")

print("\nSmoke test completed successfully.")
