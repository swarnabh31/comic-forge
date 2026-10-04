"""Draws ONE portrait or ONE panel with whichever image provider is configured.

main.py (new comics) and editor.py (chat edits) both go through here, so a panel is drawn identically
whichever path asks for it: same prompt builder, same size/steps rules, same reference preparation.

GPU_LOCK: every image provider shares one in-memory pipeline that isn't safe to call from two threads.
Anything that draws (a new comic, or a chat edit) holds this lock, so an edit made while a comic is
generating simply waits its turn.
"""
import logging
import random
import threading
from pathlib import Path

from PIL import Image, ImageOps

import config
import imagegen
import prompts
import qwen_gen
import styles

log = logging.getLogger("comicforge.render")

GPU_LOCK = threading.Lock()


def supports_references() -> bool:
    """False for the hosted API, which can't be conditioned on a character portrait."""
    return config.IMAGE_PROVIDER != "hf_inference"


def new_seed() -> int:
    return random.randint(0, 2**31 - 1)


def _engine():
    return qwen_gen if config.IMAGE_PROVIDER == "qwen" else imagegen


def _round16(v: float) -> int:
    return max(256, int(round(v / 16.0)) * 16)


def settings(quality: str = "standard") -> dict:
    """Image size / step count for a quality preset, relative to the configured base values."""
    preset = config.QUALITY_PRESETS.get(quality) or config.QUALITY_PRESETS["standard"]
    if config.IMAGE_PROVIDER == "qwen":
        base_size, base_steps = qwen_gen.BASE_SIZE, qwen_gen.BASE_STEPS
    else:
        base_size, base_steps = config.PANEL_SIZE, config.IMAGE_STEPS
    size = _round16(base_size * preset["size"])
    return {
        "size": size,
        "steps": max(4, round(base_steps * preset["steps"])),
        "portrait": _round16(size * config.PORTRAIT_SIZE_SCALE),
    }


# ----------------------------------------------------------------------------- references


def prep_reference(img: Image.Image) -> Image.Image:
    """Normalise a reference image for conditioning: apply EXIF rotation (phone photos arrive sideways),
    flatten transparency onto white, and shrink it. A 4000px original makes every panel slower and
    gains nothing: the model only needs the character's look."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("RGB")
    side = config.REF_MAX_SIDE
    if max(img.size) > side:
        img.thumbnail((side, side), Image.LANCZOS)
    return img


def open_reference(path) -> Image.Image:
    with Image.open(Path(path)) as im:
        return prep_reference(im)


# ----------------------------------------------------------------------------- drawing


def draw_portrait(name: str, appearance: str, is_person: bool, art_style, seed: int, quality: str = "standard") -> Image.Image:
    """Canonical reference portrait for a character with no uploaded picture. `name` is for logging only:
    it is deliberately NOT in the prompt (a raw name pulls the model toward a famous same-named character)."""
    st = styles.resolve(art_style)
    s = settings(quality)
    log.info("portrait %r in %s", name, st.id)
    if config.IMAGE_PROVIDER == "hf_inference":
        return imagegen.generate_hf(prompts.portrait_prompt(appearance, is_person, st, compact=True), styles.negative_prompt(st))
    compact = config.IMAGE_PROVIDER != "qwen"
    prompt = prompts.portrait_prompt(appearance, is_person, st, compact=compact)
    if config.IMAGE_PROVIDER == "qwen":
        return qwen_gen.generate(prompt, None, seed, s["portrait"], s["steps"])
    return imagegen.generate(prompt, None, seed, s["portrait"], s["steps"], styles.negative_prompt(st))


def restyle_reference(ref: Image.Image, appearance: str, is_person: bool, art_style, seed: int,
                      quality: str = "standard") -> Image.Image:
    """Redraw an UPLOADED reference (photo, anime, sketch...) in the comic's art style, keeping the character.

    This is the style-consistency fix: without it, every panel is conditioned on a picture that has its OWN
    style, which leaks into the panel and makes a comic look like it was drawn by several artists."""
    st = styles.resolve(art_style)
    s = settings(quality)
    if config.IMAGE_PROVIDER == "qwen":
        return qwen_gen.generate(prompts.restyle_prompt(appearance, is_person, st), ref, seed, s["portrait"], s["steps"])
    # SDXL: IP-Adapter at a moderate scale lets the text prompt (the style) lead while the reference sets identity.
    body = prompts.portrait_prompt(appearance, is_person, st, compact=True)
    return imagegen.generate(body, [ref], seed, s["portrait"], s["steps"], styles.negative_prompt(st), ip_scale=0.6)


def plates_enabled() -> bool:
    """Location plates (an empty picture of each room, passed to every panel in it) need an engine that takes several
    reference images natively. SDXL's IP-Adapter would BLEND the room into the character's face, so it only gets the
    text setting block and the shared seed."""
    return config.LOCATION_PLATES and config.IMAGE_PROVIDER == "qwen"


def draw_location_plate(description: str, art_style, seed: int, quality: str = "standard") -> Image.Image:
    """One empty establishing view of a location, in the comic's style."""
    st = styles.resolve(art_style)
    s = settings(quality)
    log.info("location plate in %s", st.id)
    return qwen_gen.generate(prompts.plate_prompt(description, st), None, seed, s["portrait"], s["steps"])


def draw_panel(scene_description: str, art_style, reference_images: list, seed: int, quality: str = "standard",
               meta: dict | None = None, text_size: str = "medium", location: str = "", plate: Image.Image | None = None) -> Image.Image:
    """One panel. `meta` carries the panel's camera direction: {shot, angle, placement}. `location` is the setting
    text (identical for every panel in that location); `plate` is that location's reference picture, if any."""
    st = styles.resolve(art_style)
    meta = meta or {}
    s = settings(quality)
    common = dict(shot=meta.get("shot", ""), angle=meta.get("angle", ""), placement=meta.get("placement", "center"),
                  text_size=text_size, location=location)
    if config.IMAGE_PROVIDER == "hf_inference":
        return imagegen.generate_hf(prompts.panel_prompt(scene_description, st, compact=True, **common), styles.negative_prompt(st))
    if config.IMAGE_PROVIDER == "qwen":
        prompt = prompts.panel_prompt(scene_description, st, with_refs=bool(reference_images), has_plate=plate is not None, **common)
        images = list(reference_images) + ([plate] if plate is not None else [])     # characters first, the setting LAST
        image = None if not images else images[0] if len(images) == 1 else images
        return qwen_gen.generate(prompt, image, seed, s["size"], s["steps"])
    prompt = prompts.panel_prompt(scene_description, st, compact=True, **common)
    return imagegen.generate(prompt, list(reference_images) or None, seed, s["size"], s["steps"], styles.negative_prompt(st))


# ----------------------------------------------------------------------------- warm-up


_preload_started = False
_preload_lock = threading.Lock()


def preload_async() -> None:
    """Start loading the image model in a background thread. Called at the START of a job, so the 20-30 s
    pipeline build overlaps with the LLM writing the script instead of adding to it. The model loads into
    CPU RAM first (it is CPU-offloaded), so this doesn't fight the LLM for VRAM. Safe to call repeatedly."""
    global _preload_started
    if not config.PRELOAD_IMAGE_MODEL or config.IMAGE_PROVIDER == "hf_inference":
        return
    with _preload_lock:
        if _preload_started:
            return
        _preload_started = True

    def work():
        global _preload_started
        try:
            _engine().warm()
            log.info("image model preloaded")
        except Exception:
            log.exception("image model preload failed (it will be retried on first use)")
            _preload_started = False

    threading.Thread(target=work, name="image-preload", daemon=True).start()
