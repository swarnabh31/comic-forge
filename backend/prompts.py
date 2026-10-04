"""Builds the image-model prompts.

One place, used by every image provider, following the per-panel template in the comics guide:

    [style block] [shot + angle] [scene (characters, setting, action)] [composition] [constraints]

Three things changed compared with pasting `scene, style` together:

1. The STYLE BLOCK COMES FIRST. Image models weigh early tokens most, and SDXL silently cuts anything
   after ~77 tokens, which used to remove the style entirely on long scenes.
2. The CAMERA IS DIRECTED (shot type, angle, where the subject stands), so panels vary the way a
   comic's panels should instead of being the same medium shot six times.
3. LETTERING SPACE IS RESERVED IN THE ART. Bubbles are stamped on afterwards, so the prompt asks for a calm
   band at the top of the frame, and the band is taller when the user picks a larger text size.
"""
from __future__ import annotations

import config
import styles as styles_mod

# How much calm, low-detail space to ask for in the upper frame, by text size. These are deliberately small: asking for
# a big empty band made the model shrink the subject and invent a blank wall, which wrecked scale and horizon.
# The layout engine finds calm spots on its own, so this only nudges.
HEADROOM = {"small": 0.12, "medium": 0.18, "large": 0.25}
# Tight shots have no room to spare, so they get no headroom instruction at all.
_NO_HEADROOM = ("close", "insert", "detail")

_PLACEMENT = {
    "left": "the main subject is in the left third of the frame",
    "center": "the main subject is centered in the frame",
    "right": "the main subject is in the right third of the frame",
}

_SAFE_ANGLES = {
    "eye level": "eye-level camera",
    "low angle": "slightly low camera angle",
    "high angle": "slightly high camera angle",
}


def headroom_fraction(text_size: str, shot: str = "") -> float:
    s = (shot or "").lower()
    if any(k in s for k in _NO_HEADROOM):
        return 0.0
    return HEADROOM.get(text_size, HEADROOM["medium"])


def angle_phrase(angle: str) -> str:
    """Camera angle in wording image models handle. Tilted, bird's-eye and worm's-eye views are where perspective
    breaks, so they fall back to eye level unless config.ALLOW_EXTREME_ANGLES is on."""
    a = (angle or "").strip().lower()
    if not a:
        return _SAFE_ANGLES["eye level"]
    for key, phrase in _SAFE_ANGLES.items():
        if key.split()[0] in a:
            return phrase
    if config.ALLOW_EXTREME_ANGLES:
        return a if "angle" in a or "view" in a or "tilt" in a else f"{a} angle"
    return _SAFE_ANGLES["eye level"]


def shot_phrase(shot: str, angle: str) -> str:
    shot = (shot or "").strip()
    parts = []
    if shot:
        parts.append(shot if "shot" in shot.lower() or "view" in shot.lower() else f"{shot} shot")
    parts.append(angle_phrase(angle))
    return ", ".join(parts)


def panel_prompt(scene: str, style, *, shot: str = "", angle: str = "", placement: str = "center",
                 text_size: str = "medium", with_refs: bool = False, compact: bool = False,
                 location: str = "", has_plate: bool = False) -> str:
    """Prompt for one comic panel.

    Order matters: STYLE first, then camera, then the SETTING (identical text for every panel in the same location),
    then what changes in this panel. compact=True is for CLIP-based models (SDXL): short, constraints left to the
    negative prompt."""
    st = styles_mod.resolve(style) if not hasattr(style, "prompt") else style
    cam = shot_phrase(shot, angle)
    scene = (scene or "").strip().rstrip(".")
    location = (location or "").strip().rstrip(".")
    room = f"Setting (the same room in every panel): {location}." if location else ""
    if compact:
        bits = [st.prompt, cam, location, scene, "natural perspective, correct proportions", st.lock]
        return ", ".join(b for b in bits if b)

    parts = [f"{st.prompt}.", f"{cam}.", room, f"{scene}.",
             f"Composition: {_PLACEMENT.get(placement, _PLACEMENT['center'])}."]
    pct = int(round(headroom_fraction(text_size, shot) * 100 / 5) * 5)
    if pct:
        parts.append(f"Leave some plain, low-detail space (wall, ceiling or sky) in the upper {pct}% of the frame for lettering, "
                     "without shrinking the subject or changing the room.")
    if has_plate:
        parts.append("The last reference image shows the setting. Keep the same room layout, furniture, props, window and door "
                     "positions, lighting and colors as that image, seen from this panel's camera. Do not add or move objects.")
    if with_refs:
        parts.append("Keep each character's face, hair, build and outfit colors as in their reference image(s). Draw them at natural "
                     "scale inside the setting; do not copy the reference's pose, framing or background. "
                     f"Draw everything in the {st.short} style described above, matching the style of the whole page.")
    parts.append(f"Rendering: one consistent perspective with a single horizon line, correct proportions between people and objects. "
                 f"{st.lock}. A single comic panel with no speech bubbles, caption boxes, panel borders or watermarks.")
    return " ".join(p for p in parts if p)


def plate_prompt(location: str, style, *, compact: bool = False) -> str:
    """An EMPTY establishing view of a location: the reference every panel in that location is drawn against."""
    st = styles_mod.resolve(style) if not hasattr(style, "prompt") else style
    desc = (location or "").strip().rstrip(".")
    if compact:
        return f"{st.prompt}, wide establishing shot, {desc}, empty, no people, natural perspective, {st.lock}"
    return (f"{st.prompt}. Wide establishing shot at eye level of this location: {desc}. The room is empty: no people, no characters, "
            f"no text, no signs with lettering. One consistent perspective with a single horizon line, clear layout with every major "
            f"object visible. {st.lock}.")


def portrait_prompt(appearance: str, is_person: bool, style, *, compact: bool = False) -> str:
    """Canonical reference portrait. The character's NAME is never used: a name like "Bozo" or "Mickey" is read
    as a famous character and overrides the written description."""
    st = styles_mod.resolve(style) if not hasattr(style, "prompt") else style
    body = (appearance or "").strip() or ("a single clearly defined character" if is_person else "a single cartoon object")
    if is_person:
        return (f"{st.prompt}, character reference, {body}, full body from head to feet, standing in a neutral relaxed pose, "
                f"facing the viewer, normal natural proportions, single figure, plain solid background, detailed, {st.lock}")
    return (f"{st.prompt}, a single cartoon {body}, stylized illustration, plain solid background, one object only, "
            f"no text, no labels, no callout boxes, detailed, {st.lock}")


def restyle_prompt(appearance: str, is_person: bool, style) -> str:
    """Redraw an UPLOADED reference in the comic's style, so photos/anime/sketches don't leak their own look
    into the panels."""
    st = styles_mod.resolve(style) if not hasattr(style, "prompt") else style
    who = (appearance or "").strip()
    return (f"Redraw the character in the reference image in this art style: {st.prompt}. "
            f"Keep their identity exactly: same face, hair, build, species and clothing colors"
            f"{' (' + who + ')' if who else ''}. Same character, new art style only. "
            f"{'Front-facing, full figure' if is_person else 'Single object'}, plain solid background, "
            f"no text, no labels. {st.lock}.")
