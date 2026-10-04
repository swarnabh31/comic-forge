"""Draws a whole comic: cast portraits first, then every panel, in order.

`main._run_job` used to carry three hand-copied drawing loops (hosted / Qwen / SDXL) that had drifted
apart. The worst divergence: the SDXL loop sent the RAW scene description to the image model for panels
with characters, so a cast name like "Mickey" reached the prompt and pulled the picture toward a famous
character: exactly the bug `llm.rewrite_scene` exists to prevent, fixed in the other two loops only.
One loop, one behaviour:

  1. every cast member gets ONE reference image, used for the whole comic
       - generated from their written appearance, or
       - the user's upload, redrawn ONCE in the comic's art style (see render.restyle_reference), so the
         look of a photo or an anime picture can't leak into the panels
  2. every LOCATION gets one empty reference picture (the set) and one shared seed
  3. every panel is drawn from the setting text + the name-free scene + its camera direction, conditioned on the
     references of the characters in it, with the style block first in the prompt
  3. each finished panel is reported immediately (`on_panel`) so the UI can show it while the rest are drawn
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

import config
import llm
import render
import styles
from schemas import ComicScript

log = logging.getLogger("comicforge.pipeline")


@dataclass
class DrawResult:
    panel_images: list = field(default_factory=list)
    refs: dict = field(default_factory=dict)             # name -> [image used for conditioning]
    plates: dict = field(default_factory=dict)           # location id -> empty establishing picture
    seeds: list = field(default_factory=list)            # the seed each panel was drawn with
    originals: dict = field(default_factory=dict)        # name -> the user's own picture (uploads only)
    warnings: list = field(default_factory=list)


def build_identities(script: ComicScript, uploaded_by_name: dict, model: str) -> dict[str, str]:
    """Canonical, NAME-FREE description per cast member, used to rewrite every scene before it reaches the
    image model. An upload's description (typed by the user, or read from the picture by a vision model)
    always wins over whatever the script writer invented."""
    identities: dict[str, str] = {}
    for name, up in uploaded_by_name.items():
        if not up.description and up.reference_images:
            derived = llm.describe_reference(up.reference_images[0], model)   # safety net; normally set at upload
            if derived:
                up.description = derived
        if up.description:
            identities[name] = up.description
    for member in script.cast:
        if member.name in identities:
            member.appearance = identities[member.name]
        elif member.appearance:
            identities[member.name] = member.appearance
    return identities


def draw_comic(
    script: ComicScript,
    art_style: str,
    quality: str,
    text_size: str,
    uploaded_by_name: dict,
    identities: dict[str, str],
    seed: int,
    progress: Callable[[str, float], None],
    on_panel: Optional[Callable[[int, Image.Image], None]] = None,
) -> DrawResult:
    st = styles.resolve(art_style)
    out = DrawResult()
    use_refs = render.supports_references()
    members = {m.name: m for m in script.cast}
    normalize = config.STYLE_NORMALIZE_REFS

    # ---- 1. reference images ------------------------------------------------------------------
    plan: list[tuple[str, str]] = []                      # (name, "restyle" | "portrait" | "as-is")
    if use_refs:
        for name in dict.fromkeys([*uploaded_by_name, *members]):
            up = uploaded_by_name.get(name)
            if up and up.reference_images:
                plan.append((name, "restyle" if normalize else "as-is"))
            elif name in members:
                plan.append((name, "portrait"))
    # Locations actually used by a panel. Each gets ONE plate, and ONE seed shared by all its panels: panels drawn
    # from the same seed start from the same layout, so the room stops rearranging itself between panels.
    locs = {l.id: l for l in script.locations}
    used = list(dict.fromkeys(p.location for p in script.panels if p.location in locs))
    loc_seed = {lid: seed + 101 * (n + 1) for n, lid in enumerate(used)}
    loc_text = {lid: llm.rewrite_scene(locs[lid].description, identities) for lid in used}
    make_plates = render.plates_enabled() and bool(used)
    work = sum(1 for _, how in plan if how != "as-is") + len(script.panels) + (len(used) if make_plates else 0)
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        progress(msg, 0.10 + 0.80 * (done / max(work, 1)))
        done += 1

    for name, how in plan:
        member = members.get(name)
        appearance = identities.get(name) or (member.appearance if member else "")
        is_person = member.is_person if member else True
        if how == "portrait":
            tick(f"Sketching {name}...")
            out.refs[name] = [render.draw_portrait(name, appearance, is_person, st, seed, quality)]
            continue
        original = render.open_reference(uploaded_by_name[name].reference_images[0])
        out.originals[name] = original
        out.refs[name] = [original]
        if how == "restyle":
            tick(f"Redrawing {name} in {st.label} style...")
            try:
                out.refs[name] = [render.restyle_reference(original, appearance, is_person, st, seed, quality)]
            except Exception:
                # A failed restyle must not lose the comic: fall back to the user's own picture.
                log.exception("restyle failed for %s; using the original picture", name)
                out.warnings.append(f"Couldn't restyle {name}'s picture, so it was used as uploaded; the style may vary in their panels.")

    # ---- 2. location plates ---------------------------------------------------------------------
    if make_plates:
        for lid in used:
            tick(f"Building the set: {lid.replace('_', ' ')}...")
            try:
                out.plates[lid] = render.draw_location_plate(loc_text[lid], st, loc_seed[lid], quality)
            except Exception:
                log.exception("location plate failed for %s; drawing its panels from text only", lid)
                out.warnings.append(f"Couldn't draw a reference for the {lid.replace('_', ' ')} set, so its panels rely on the text description only.")

    # ---- 3. panels ----------------------------------------------------------------------------
    for i, panel in enumerate(script.panels):
        tick(f"Drawing panel {i + 1} of {len(script.panels)}...")
        names = list(dict.fromkeys(panel.characters))
        panel_refs = [img for n in names for img in out.refs.get(n, [])] if use_refs else []
        scene = llm.rewrite_scene(panel.scene_description, identities)
        meta = {"shot": panel.shot, "angle": panel.angle, "placement": panel.placement}
        panel_seed = loc_seed.get(panel.location, seed + i + 1)
        img = render.draw_panel(scene, st, panel_refs, panel_seed, quality, meta, text_size,
                                loc_text.get(panel.location, ""), out.plates.get(panel.location))
        out.seeds.append(panel_seed)
        out.panel_images.append(img)
        if on_panel:
            on_panel(i, img)
    return out
