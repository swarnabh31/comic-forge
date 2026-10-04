"""SDXL + IP-Adapter engine (IMAGE_PROVIDER=local) and the hosted HF engine (IMAGE_PROVIDER=hf_inference).

One low-level call each; prompt wording and negative prompts come from prompts.py / styles.py. A single
seed per comic (plus a small per-panel offset) is chosen by the caller so line weight and palette don't
reset from panel to panel.

If SDXL + IP-Adapter isn't enough for identity consistency, the Qwen engine (qwen_gen.py) conditions on
reference images natively and is the stronger option.
"""
import threading
from io import BytesIO

from PIL import Image

import config

IP_ADAPTER_SCALE = 0.85

_pipe = None
_ip_adapter_ready = False
_LOAD_LOCK = threading.Lock()


def _load_pipeline():
    global _pipe
    with _LOAD_LOCK:
        if _pipe is not None:
            return _pipe
        import torch
        from diffusers import StableDiffusionXLPipeline

        _pipe = StableDiffusionXLPipeline.from_pretrained(
            config.SD_MODEL_ID,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
            variant="fp16" if config.DEVICE == "cuda" else None,
        ).to(config.DEVICE)
        # Not enable_attention_slicing(): it conflicts with load_ip_adapter() in current diffusers.
        # On an 8-16 GB card use enable_model_cpu_offload() instead, which doesn't conflict.
        return _pipe


def _ensure_ip_adapter():
    global _ip_adapter_ready
    pipe = _load_pipeline()
    with _LOAD_LOCK:
        if not _ip_adapter_ready:
            pipe.load_ip_adapter(config.IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=config.IP_ADAPTER_WEIGHT_SDXL)
            _ip_adapter_ready = True
    return pipe


def warm() -> None:
    _ensure_ip_adapter()


def _seeded_generator(seed: int):
    import torch
    return torch.Generator(device=config.DEVICE).manual_seed(seed)


def new_comic_seed() -> int:
    import random
    return random.randint(0, 2**31 - 1)


def generate(prompt: str, refs: list | None = None, seed: int = 0, size: int | None = None,
             steps: int | None = None, negative: str | None = None, ip_scale: float = IP_ADAPTER_SCALE) -> Image.Image:
    """Text-to-image, or reference-conditioned (IP-Adapter) when `refs` is given.

    Several references are BLENDED by IP-Adapter, not independently controlled; fidelity is best with 1-2."""
    size = size or config.PANEL_SIZE
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative or "",
        num_inference_steps=steps or config.IMAGE_STEPS,
        height=size,
        width=size,
        generator=_seeded_generator(seed),
    )
    if refs:
        pipe = _ensure_ip_adapter()
        pipe.set_ip_adapter_scale(ip_scale)
        # One list per loaded IP-Adapter (we load one), holding every image to blend: hence the extra [ ].
        kwargs["ip_adapter_image"] = [list(refs)]
    else:
        pipe = _load_pipeline()
        if _ip_adapter_ready:                 # adapter is installed: switch it off instead of unloading it
            pipe.set_ip_adapter_scale(0.0)
            kwargs["ip_adapter_image"] = [Image.new("RGB", (64, 64))]
    return pipe(**kwargs).images[0]


def generate_hf(prompt: str, negative: str | None = None) -> Image.Image:
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=config.HF_IMAGE_MODEL, token=config.HF_TOKEN)
    return client.text_to_image(prompt, negative_prompt=negative or None)


def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
