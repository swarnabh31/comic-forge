"""Turns a one-line idea into a structured comic script (title + panels).

Two interchangeable providers:
- Ollama (local): a JSON schema is passed as `format` so the model is
  constrained to return valid, parseable JSON. Model list is read live from
  `ollama list` so the UI can offer a dropdown instead of a hardcoded name.
- Hugging Face Inference API: same prompt, sent as a chat completion.
"""
import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path

import httpx

from PIL import Image

import config
import expert
import styles
from schemas import ComicScript

log = logging.getLogger("comicforge.llm")

SYSTEM_PROMPT = """You are a professional comic book writer and storyboard artist. Turn the user's idea into a \
short comic script with exactly {PANEL_COUNT} panels.

{EXPERT}
CAST RULES
You are given a list of user-provided characters (name + description). These are the CORE cast of the comic: \
include every one of them in the cast and in the panels where they fit. Use their exact names in dialogue and \
panel references. You may introduce minor unnamed characters only if the story requires them; do not invent new \
named characters that overlap with a user-provided one.

For each user-provided character, USE their description verbatim in the cast. Do not add, remove, or change \
features the user did not list. The description is the canonical identity of the character for the artist.

For any character the user did NOT provide (you are introducing them), write a concrete visual appearance (subject \
type, face/key features, hair/fur, build, exact clothing and colors) an artist could draw from without seeing \
anything else, and mark is_person true for a human/animal or false for an object. This includes non-human things \
with a speaking role (a talking fridge, a robot, a car).

NON-HUMAN CHARACTERS (is_person=false): describe them as a **stylized cartoon** matching the art style of the \
humans in the same comic. Avoid realistic material descriptions (brushed steel, stainless finish), brand/spec \
language (model number, cubic feet) or anything that reads as a product catalogue. A fridge with personality has \
a face on the screen, a little smile, cartoon proportions.

LOCATIONS (this is what keeps the environment identical from panel to panel)
Define every place the story happens in ONCE, in "locations": at most {MAX_LOCATIONS}, and use as few as the \
story allows (one is best). Each has a short "id" (e.g. "boardroom") and a "description" that fixes the room: its \
shape, the large fixed objects and exactly where they are ("long wooden table down the middle, projector screen on \
the back wall, podium front-left, windows on the right wall"), lighting and time of day, and the colour scheme. \
Write the description as if it were a set that stays built for the whole comic. Every panel gets a "location" id.

PANEL RULES
Each panel needs: a camera direction ("shot" and "angle"), a "location" id, a scene_description, an optional short \
caption, and optional dialogue.
- shot is one of: wide establishing, full, medium, close-up, extreme close-up, insert, over-the-shoulder.
- angle is almost always "eye level". Use "low angle" or "high angle" only occasionally. No tilted or bird's-eye \
angles: the artist cannot draw them reliably.
- scene_description describes ONLY what changes in this panel: who is where in the shot, their pose, their specific \
expression, and any prop they handle. Do NOT re-describe the room (the location already does that) and do not \
invent new furniture or props that are not in the location. Never include dialogue, captions or lettering.
- Keep each character's position and facing direction consistent between neighbouring panels in the same location.

TEXT BUDGET (the user chose {SIZE_NAME} lettering, so there is limited room in each panel): every dialogue line is \
at most {WORDS} words; a caption is at most {CAPWORDS} words; at most 2 dialogue lines per panel. A caption is \
for narration or scene-setting ONLY: it must never repeat or paraphrase a line that is also in that panel's \
dialogue.

SPEAKER / PANEL RULE (strict): every panel has dialogue from AT MOST ONE speaker, and that speaker is the clear \
visual focus of the panel (the camera is aimed at them). The `characters` field for a panel lists ONLY that \
speaker (or is empty for a panel with no characters), even if another character is also physically present in the \
scene. If a conversation goes back and forth, alternate one speaker per panel.

Respond with ONLY JSON matching this shape:
{"title": "string", "cast": [{"name": "string", "appearance": "string", "is_person": true}], \
"locations": [{"id": "string", "description": "string"}], \
"panels": [{"index": 1, "shot": "string", "angle": "string", "location": "location id", \
"scene_description": "string", "characters": ["Name"], "caption": "string or null", \
"dialogue": [{"speaker": "Name", "text": "string", "type": "speech or thought"}]}]}"""


def build_system_prompt(panel_count: int, text_size: str = "medium") -> str:
    """The script writer's instructions: craft knowledge + cast rules + the text budget for the chosen size."""
    size = text_size if text_size in expert.WORD_LIMITS else "medium"
    expert_block = expert.craft_rules(panel_count)
    return (
        SYSTEM_PROMPT.replace("{EXPERT}", (expert_block + "\n") if expert_block else "")
        .replace("{PANEL_COUNT}", str(panel_count))
        .replace("{MAX_LOCATIONS}", str(config.MAX_LOCATIONS))
        .replace("{SIZE_NAME}", size.upper())
        .replace("{WORDS}", str(expert.word_limit(size)))
        .replace("{CAPWORDS}", str(expert.caption_word_limit(size)))
    )


def _build_user_prompt(idea: str, characters: list[dict]) -> str:
    if not characters:
        cast = "No fixed cast is provided; invent the cast you need (a person, animal, or non-human thing with a role)."
    else:
        cast = "Cast provided by the user (these are the stars of the comic):\n"
        for c in characters:
            cast += f'- {c["name"]}: {c["description"] or "(no description; describe them from the scene)"}\n'
        cast += "Use these cast members by name and description in the panels, and include every one of them."
    return f"Story idea: {idea}\n{cast}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """First complete JSON object in the model's output.

    Reasoning models (qwen3 and friends) may emit a <think>...</think> block, and models like to wrap JSON in
    prose or code fences. The old greedy `{.*}` regex grabbed from the first brace to the last one anywhere in the
    text, which broke on both; this walks candidate start positions and lets the JSON decoder find the end."""
    text = _THINK_RE.sub("", text or "").strip()
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"No JSON object found in model output: {text[:300]}")


def list_ollama_models() -> list[str]:
    resp = httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def pick_vision_model() -> str | None:
    """Pick a vision-capable Ollama model from the ones actually installed on the
    machine. Prefers the model configured as the default LLM if it supports
    vision; otherwise walks the installed list. Returns None if no installed
    model supports vision (the caller then skips the auto-describe step).
    """
    if config.LLM_PROVIDER != "ollama":
        return None
    try:
        models = list_ollama_models()
    except Exception:
        return None
    for m in models:
        if supports_vision(m):
            return m
    return None


def _ollama_options(temperature: float) -> dict:
    # num_ctx matters: Ollama's default window is small and it truncates the START of an over-long prompt,
    # which silently deletes the instructions.
    return {"temperature": temperature, "num_ctx": config.OLLAMA_NUM_CTX}


def _generate_with_ollama(system: str, prompt: str, model: str, fmt) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "format": fmt,                         # a JSON schema (constrained decoding) or "json"
        "stream": False,
        "keep_alive": config.LLM_KEEP_ALIVE,
        "options": _ollama_options(0.8),
    }
    resp = httpx.post(f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return _extract_json(resp.json()["response"])


def _generate_with_hf(system: str, prompt: str) -> dict:
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{config.HF_LLM_MODEL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
        json={
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": 2500,
            "temperature": 0.8,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return _extract_json(resp.json()["choices"][0]["message"]["content"])


def unload(model: str) -> None:
    """Tell Ollama to drop `model` from VRAM now. The image model needs that memory next, and by default Ollama
    keeps the LLM resident for minutes, so the two fight over the GPU. Best effort; never raises."""
    if config.LLM_PROVIDER != "ollama" or not model:
        return
    try:
        httpx.post(f"{config.OLLAMA_HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
    except Exception:
        pass


def _dedupe_captions(script_dict: dict) -> None:
    """Safety net: drop a caption if it's essentially the same text as a dialogue
    line in the same panel (the LLM sometimes echoes a line into both fields)."""
    for panel in script_dict.get("panels", []):
        caption = (panel.get("caption") or "").strip().lower()
        if not caption:
            continue
        for line in panel.get("dialogue", []):
            spoken = line.get("text", "").strip().lower()
            if spoken and (caption == spoken or caption in spoken or spoken in caption):
                panel["caption"] = None
                break


def _normalize_speaker(script_dict: dict) -> int:
    """Enforce one speaker per panel. Returns how many dialogue lines had to be dropped to do it.

    Several characters in one panel get blended into one face by the image model, so a panel has at most one
    speaker and panel.characters == [that speaker]. The speaker is the visual focus, as in a real comic."""
    dropped = 0
    for panel in script_dict.get("panels", []):
        dialogue = panel.get("dialogue", [])
        if not dialogue:
            continue
        primary = dialogue[0].get("speaker")
        kept = [line for line in dialogue if line.get("speaker") == primary]
        dropped += len(dialogue) - len(kept)
        panel["dialogue"] = kept
        if primary:
            panel["characters"] = [primary]
    return dropped


_NULLISH = {"", "none", "null", "n/a", "na", "nil", "no caption"}


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_as_text(x) for x in v).strip()
    return str(v).strip()


def _sanitize_script(raw: dict, panel_count: int) -> tuple[dict, list[str]]:
    """Coerce what a language model actually returns into what ComicScript accepts, and report anything lost.

    Models return a caption as a list, "null" as a string, a dialogue line as a bare string, a speaker that isn't
    in the cast, a panel with no scene... One of these used to fail the whole job after the user had waited."""
    notes: list[str] = []
    cast = []
    for m in raw.get("cast") or []:
        if not isinstance(m, dict) or not _as_text(m.get("name")):
            continue
        cast.append({
            "name": _as_text(m["name"])[:40],
            "appearance": _as_text(m.get("appearance")),
            "is_person": m.get("is_person") not in (False, "false", "False", 0, "no"),
        })
    by_lower = {c["name"].lower(): c["name"] for c in cast}

    # Location bible: unique slug ids, capped (every extra location is another room the artist must keep consistent).
    locations, seen = [], set()
    for loc in raw.get("locations") or []:
        if not isinstance(loc, dict) or not _as_text(loc.get("description")):
            continue
        lid = re.sub(r"[^a-z0-9]+", "_", _as_text(loc.get("id") or loc.get("name")).lower()).strip("_")[:30] or f"place_{len(locations) + 1}"
        if lid in seen:
            continue
        if len(locations) >= config.MAX_LOCATIONS:
            notes.append(f"The script used more than {config.MAX_LOCATIONS} locations; extra ones were merged into the others to keep the environment consistent.")
            break
        seen.add(lid)
        locations.append({"id": lid, "description": _as_text(loc["description"])})

    def slug(v) -> str:
        return re.sub(r"[^a-z0-9]+", "_", _as_text(v).lower()).strip("_")[:30]

    panels = []
    for item in (raw.get("panels") or []):
        if not isinstance(item, dict) or not _as_text(item.get("scene_description")):
            continue
        dialogue = []
        for line in item.get("dialogue") or []:
            if isinstance(line, str):
                line = {"speaker": "", "text": line}
            if not isinstance(line, dict) or not _as_text(line.get("text")):
                continue
            speaker = _as_text(line.get("speaker"))
            dialogue.append({"speaker": by_lower.get(speaker.lower(), speaker), "text": _as_text(line["text"]),
                             "type": _as_text(line.get("type")) or "speech"})
        chars = item.get("characters") or []
        if isinstance(chars, str):
            chars = [chars]
        known = []
        for c in chars:
            canon = by_lower.get(_as_text(c).lower())
            if canon and canon not in known:
                known.append(canon)
        caption = _as_text(item.get("caption"))
        if len(dialogue) > 3:
            notes.append(f"Panel {len(panels) + 1} had {len(dialogue)} dialogue lines; only the first 3 fit.")
        panels.append({
            "index": len(panels) + 1,
            "scene_description": _as_text(item["scene_description"]),
            "characters": known,
            "caption": None if caption.lower() in _NULLISH else caption,
            "dialogue": dialogue[:3],
            "shot": _as_text(item.get("shot")),
            "angle": _as_text(item.get("angle")),
            # The speaker is drawn centered. A model-chosen left/right made the character jump around the frame from
            # panel to panel; placement can still be set deliberately through the chat assistant.
            "placement": "center",
            "location": slug(item.get("location")),
        })
    # A panel with a missing or unknown location stays in the previous panel's room (scenes are continuous).
    ids = {l["id"] for l in locations}
    prev = locations[0]["id"] if locations else ""
    for p in panels:
        if p["location"] not in ids:
            p["location"] = prev
        prev = p["location"]
    if panels and not locations:
        notes.append("The script defined no locations, so the environment may vary between panels. Try again or use a larger model.")
    panels = panels[:panel_count]
    for i, p in enumerate(panels, start=1):
        p["index"] = i
    if not panels:
        raise ValueError("The model's script had no usable panels.")
    if len(panels) < panel_count:
        notes.append(f"The model wrote {len(panels)} panels instead of the {panel_count} you asked for.")
    used = {p["location"] for p in panels}
    locations = [l for l in locations if l["id"] in used]
    return {"title": _as_text(raw.get("title")) or "Untitled Comic", "cast": cast, "locations": locations, "panels": panels}, notes


def _strict_schema() -> dict:
    """ComicScript's JSON schema with EVERY property required. Pydantic only requires fields without defaults, so
    under constrained decoding a model could otherwise leave out shot / angle / placement / caption entirely."""
    import copy
    schema = copy.deepcopy(ComicScript.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def generate_script_with_notes(idea: str, model: str, panel_count: int, characters: list[dict],
                               text_size: str = "medium") -> tuple[ComicScript, list[str]]:
    """Write the script. Returns (script, notes about anything that had to be repaired or dropped).

    First try uses the JSON schema as Ollama's `format` (constrained decoding: the output cannot be malformed).
    If that fails for any reason (an older Ollama, a model that doesn't cope with grammars, a script with no
    usable panels) it retries once with plain JSON mode and the failure reason in the prompt."""
    system = build_system_prompt(panel_count, text_size)
    prompt = _build_user_prompt(idea, characters)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            if config.LLM_PROVIDER == "ollama":
                fmt = _strict_schema() if attempt == 0 else "json"
                raw = _generate_with_ollama(system, prompt, model, fmt)
            else:
                raw = _generate_with_hf(system, prompt)
            clean, notes = _sanitize_script(raw, panel_count)
            _dedupe_captions(clean)
            dropped = _normalize_speaker(clean)
            if dropped:
                notes.append(f"{dropped} dialogue line(s) from a second speaker were dropped (one speaker per panel).")
            return ComicScript.model_validate(clean), notes
        except Exception as exc:                       # network, HTTP 400 for the schema, bad JSON, no panels...
            last_error = exc
            log.warning("script attempt %d failed: %s", attempt + 1, exc)
            prompt = _build_user_prompt(idea, characters) + (
                f"\n\n(Your previous answer could not be used: {str(exc)[:200]}. "
                "Answer again with ONLY the JSON object, with every panel having a scene_description.)")
    raise RuntimeError(f"The language model couldn't produce a usable script: {last_error}")


def generate_script(idea: str, model: str, panel_count: int, characters: list[dict], text_size: str = "medium") -> ComicScript:
    return generate_script_with_notes(idea, model, panel_count, characters, text_size)[0]


# --- Identity helpers (the fix that stops the "invented characters" problem) --
#
# A cast member's raw name (e.g. "Mickey", "Bozo") is a very strong prompt token
# for an image model — "Mickey" almost always resolves to Mickey Mouse and
# "Bozo" to a clown — so the name ends up overriding the reference image. The
# fix is to NEVER put the raw name into the image prompt. Instead we build a
# literal, name-free identity description once (from the upload, via vision),
# store it, and let `rewrite_scene` swap the name out of every scene before it
# reaches the diffuser. The image model then sees only the literal appearance
# plus the reference image, which is exactly who the user meant.


def describe_reference(image_path: str | Path, model: str) -> str:
    """Look at a character's uploaded image with the (vision) LLM and write a
    literal, name-free description of it — subject type, face, hair/fur, build,
    exact clothing and colors. Returns "" if the model has no vision or fails,
    so callers fall back to the user's manual description.
    """
    if not supports_vision(model):
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((640, 640))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

    sys = ("You are a precise art-director describing a reference picture so another artist can "
           "replicate the SAME character. Read only what is visible; do not guess a famous "
           "character by name or add anything not in the picture.")
    user = ("Describe the single character in this reference image as one plain sentence an "
            "illustrator could draw from: state the subject type (person / animal / thing); the "
            "exact species or item if an animal/thing; face and key features; hair or fur; "
            "build and proportions; and the exact clothing and colors. "
            "Do NOT include the character's name. Do NOT invent a real-world identity. "
            "Output the description text only, no heading, no quotes.")
    try:
        return _chat_text(sys, user, model, [b64]).strip()
    except Exception:
        return ""


def rewrite_scene(scene_description: str, identities: dict[str, str]) -> str:
    """Replace every occurrence of a cast name in `scene_description` with its literal
    identity description, so the raw name never reaches the image prompt.

    `identities` maps canonical name -> name-free description (e.g.
    {"Mickey": "a small toddler girl with black curly hair and a red polka-dot bib"}).
    Names are matched case-insensitively on word boundaries, longest-name-first so a
    longer name is not clobbered by a shorter one that is a substring of it. Names with no
    description are left untouched (they fall through to the reference image)."""
    if not identities or not scene_description:
        return scene_description
    # Longest first to avoid a short name eating part of a longer name.
    ordered = sorted(identities.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = scene_description
    for name, desc in ordered:
        if not desc:
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        out = pattern.sub(desc, out)
    return out


# --- Chat assistant (comic editing) ------------------------------------------
_VISION_CACHE: dict[str, bool] = {}


def supports_vision(model: str) -> bool:
    """True if this Ollama model can look at images (asked via /api/show). Only a
    successful answer is cached, so starting Ollama later doesn't leave it stuck on False."""
    if config.LLM_PROVIDER != "ollama" or not model:
        return False
    if model not in _VISION_CACHE:
        try:
            resp = httpx.post(f"{config.OLLAMA_HOST}/api/show", json={"model": model}, timeout=10)
            resp.raise_for_status()
            _VISION_CACHE[model] = "vision" in (resp.json().get("capabilities") or [])
        except Exception:
            return False
    return _VISION_CACHE[model]


def _chat_ollama(system: str, messages: list[dict], model: str, images: list[str] | None) -> str:
    msgs = [{"role": "system", "content": system}] + [dict(m) for m in messages]
    if images:
        for m in reversed(msgs):
            if m["role"] == "user":
                m["images"] = images
                break
    payload = {"model": model, "messages": msgs, "format": "json", "stream": False,
               "keep_alive": config.LLM_KEEP_ALIVE, "options": _ollama_options(0.4)}
    resp = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _chat_hf(system: str, messages: list[dict]) -> str:
    resp = httpx.post(
        f"https://api-inference.huggingface.co/models/{config.HF_LLM_MODEL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.HF_TOKEN}"},
        json={"messages": [{"role": "system", "content": system}] + messages, "max_tokens": 2000, "temperature": 0.4},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _chat_text(system: str, user: str, model: str, images: list[str] | None = None) -> str:
    """One plain-text (non-JSON) turn. Used for describing a reference image —
    Ollama only, since image input is the point and the HF path here is text-only."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if images:
        msgs[-1]["images"] = images
    payload = {"model": model, "messages": msgs, "stream": False,
               "keep_alive": config.LLM_KEEP_ALIVE, "options": _ollama_options(0.2)}
    resp = httpx.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def chat_json(system: str, messages: list[dict], model: str, images: list[str] | None = None) -> dict:
    """One chat turn that must come back as a JSON object. If the model wraps it in
    prose or breaks the syntax, it gets exactly one chance to repair it."""
    for attempt in range(2):
        text = _chat_ollama(system, messages, model, images) if config.LLM_PROVIDER == "ollama" else _chat_hf(system, messages)
        try:
            return _extract_json(text)
        except ValueError:      # includes json.JSONDecodeError
            if attempt:
                raise
            messages = messages + [
                {"role": "assistant", "content": text[:2000]},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
            ]
