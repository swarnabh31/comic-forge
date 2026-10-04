"""Qwen-Image-2.1 engine: one low-level call. Prompt wording lives in prompts.py.

Identity strategy: Qwen-Image-2.1 takes reference images through its native image conditioning (the
`image=` argument), which holds non-facial objects (fridge, robot) far better than an IP-Adapter blend.
The model is large (~33 GB bf16) so `enable_model_cpu_offload()` is used to fit a 32 GB card.
"""
import logging
import threading
from io import BytesIO

from PIL import Image

import config

log = logging.getLogger("comicforge.qwen")

_QWEN_MODEL_ID = getattr(config, "QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1")
BASE_STEPS = int(getattr(config, "QWEN_STEPS", 20))
BASE_SIZE = int(getattr(config, "QWEN_PANEL_SIZE", 1024))

_pipe = None
_LOAD_LOCK = threading.Lock()          # a background preload and a real request must never load twice


def _load_pipeline():
    global _pipe
    with _LOAD_LOCK:
        if _pipe is not None:
            return _pipe
        import torch
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(_QWEN_MODEL_ID, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        _pipe = pipe
        return pipe


def warm() -> None:
    _load_pipeline()


def _seeded_generator(seed: int):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(seed)


def new_comic_seed() -> int:
    import random
    return random.randint(0, 2**31 - 1)


def generate(prompt: str, image=None, seed: int = 0, size: int | None = None, steps: int | None = None,
             negative: str | None = None) -> Image.Image:
    """`image` may be one PIL image or a list of them (reference conditioning). `negative` is accepted for
    interface parity with the SDXL engine and ignored: this pipeline is steered with positive wording."""
    pipe = _load_pipeline()
    size = size or BASE_SIZE
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps or BASE_STEPS,
        height=size,
        width=size,
        generator=_seeded_generator(seed),
    )
    if image is not None:
        kwargs["image"] = image
    return pipe(**kwargs).images[0]


def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
