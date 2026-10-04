"""Comic-craft knowledge for the language model.

`knowledge/comics_expert.md` is the full guide. It is written as a standalone system prompt for an
interactive assistant, so it is not pasted blindly into this app's prompts: Comic Forge needs the
model to answer with ONE strict JSON object, while the guide describes conversational output formats.

Three modes (config.COMICS_EXPERT):

  condensed  (default) The parts of the guide that change what a script-writer should DO, distilled
             into ~900 tokens: story arc, drawable moments, shot variety, word/image economy,
             lettering awareness. Cheap enough for every call.
  full       The entire guide, framed as reference knowledge, with the guide's own output formats
             explicitly overridden. ~6.5k tokens: needs a large context window (OLLAMA_NUM_CTX).
  off        Nothing (the original behaviour).

The chat assistant always gets the short `editing_rules()`: it runs on every message, and edits
are small, so it needs the guardrails rather than the whole book.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import config

GUIDE_PATH = Path(__file__).parent / "knowledge" / "comics_expert.md"

# Words per balloon, by lettering size. Bigger lettering means less room, so the writer is told to
# be tighter BEFORE the layout engine ever has to shrink anything.
WORD_LIMITS = {"small": 14, "medium": 12, "large": 9}
CAPTION_WORD_LIMITS = {"small": 18, "medium": 15, "large": 12}


def word_limit(text_size: str) -> int:
    return WORD_LIMITS.get(text_size, WORD_LIMITS["medium"])


def caption_word_limit(text_size: str) -> int:
    return CAPTION_WORD_LIMITS.get(text_size, CAPTION_WORD_LIMITS["medium"])


@lru_cache(maxsize=1)
def full_guide() -> str:
    try:
        return GUIDE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def arc_guidance(panel_count: int) -> str:
    """Story shape for this many panels (McCloud / Cohn narrative grammar, kishotenketsu, three-act)."""
    if panel_count <= 2:
        return "two beats: SETUP, then a PUNCHLINE or reversal that recontextualises the setup."
    if panel_count == 3:
        return "three beats: SETUP, ESCALATION, PUNCHLINE."
    if panel_count == 4:
        return ("4-koma / kishotenketsu rhythm: panel 1 SETUP (who, where, what they want), panel 2 DEVELOPMENT, "
                "panel 3 TWIST, panel 4 PUNCHLINE or emotional landing.")
    if panel_count <= 6:
        return ("Establisher (who/where) -> Initial (the action starts) -> Prolongation (tension or comedy builds) "
                "-> PEAK (the single most dramatic or funniest moment) -> Release (the aftermath, ending on a hook).")
    return ("three acts: SETUP (first 1-2 panels: who, where, what they want), CONFRONTATION (obstacles escalate; "
            "include one quiet or wordless panel for rhythm), RESOLUTION (PEAK, then Release ending on a hook).")


_CONDENSED = """\
COMIC CRAFT RULES (professional comics practice; apply them):
STORY
- Think in panels, not prose. Every panel shows ONE clear, drawable moment that changes something: information, emotion, position or tension. If a panel changes nothing, cut or merge it.
- Never script what a still picture cannot show ("he remembers his childhood"). Turn it into a visible action, object or facial expression.
- Story shape for {PANELS} panels: {ARC}
- Make the PEAK the most visually emphasised panel (close-up, low angle or dramatic wide). End on a hook, reversal or punchline.
- Skip moments on purpose. The reader fills the gap between panels, so do not draw every step.
CAMERA AND STAGING
- Vary shot and angle. Never three similar talking-head shots in a row. Mix: wide/establishing, medium, close-up, extreme close-up (eyes, hands), insert (a key object), over-the-shoulder.
- Open with a wide or establishing shot so the reader knows where we are. Camera is eye level almost always; a mild low angle (power) or high angle (vulnerability) now and then. Never tilted or bird's-eye views.
- Keep continuity: same costume and props, and each character on the same side of the frame throughout a conversation (180-degree rule).
- SETTING CONTINUITY: define each location ONCE (layout, fixed props and where they stand, lighting) and reuse it. Use 1-3 locations. A panel's scene_description then states only what changes: pose, a SPECIFIC facial expression (what the character is hiding or wants, not just "sad"), and props being handled.
WORDS AND PICTURES
- Words add what the picture cannot. Never describe what is visible and never narrate an obvious emotion. A panel may be silent.
- Captions carry time/place, narration or inner voice only. Dialogue is compressed speech: one idea per balloon, and each character has a distinct voice.
LETTERING AWARENESS
- Balloons are stamped onto the finished art afterwards and are placed automatically over calm areas, so keep dialogue short and never describe lettering in a scene.
- The first speaker reads first (left/top); a responder sits right/lower.
CHARACTERS
- Give every character one or two anchor features (a hair tuft, hat, scar, signature colour) and keep them identical in every description. Use simple iconic faces for the protagonist and richer detail in backgrounds.
"""

_EDITING = """\
CRAFT RULES FOR EDITS (professional comics practice):
- Each panel shows ONE drawable moment. Do not script what a picture cannot show; turn it into an action or expression.
- Vary shot and angle between neighbouring panels (wide, medium, close-up, insert). Do not repeat the same talking-head shot.
- Words add what the picture cannot: no describing visible things, no narrating obvious emotion, captions only for time/place/narration.
- Reuse the comic's existing locations (their ids are listed in CURRENT COMIC); do not invent new furniture, and keep angles at eye level unless a mild low/high angle is clearly needed.
- A new or rewritten panel must keep character looks, costumes and the story's arc intact (an ending panel should land the hook or punchline).
"""


def craft_rules(panel_count: int, mode: str | None = None) -> str:
    """Knowledge block for the script writer."""
    mode = (mode or config.COMICS_EXPERT or "condensed").lower()
    if mode == "off":
        return ""
    if mode == "full" and full_guide():
        return (
            "COMICS CRAFT GUIDE (reference knowledge). Apply it to the script you write. IMPORTANT: this guide "
            "also describes conversational answer formats (page scripts, critiques, thumbnails, section 11 and 15). "
            "Ignore those: your ONLY output is the JSON object described at the end of these instructions.\n"
            "Story shape for this comic: " + arc_guidance(panel_count) + "\n"
            "<<<GUIDE\n" + full_guide() + "\nGUIDE>>>\n"
        )
    return _CONDENSED.replace("{PANELS}", str(panel_count)).replace("{ARC}", arc_guidance(panel_count))


def editing_rules() -> str:
    return "" if (config.COMICS_EXPERT or "").lower() == "off" else _EDITING


# --- Deterministic checks (section 12 of the guide, the parts that can be checked without a model) ----------------


def qa_script(script, text_size: str = "medium") -> list[str]:
    """Quality-checklist findings about a script, as short user-facing notes. Never raises."""
    notes: list[str] = []
    try:
        limit = word_limit(text_size)
        panels = script.panels
        long_lines = [
            i for i, p in enumerate(panels, 1) for d in p.dialogue if len(d.text.split()) > limit + 4
        ]
        if long_lines:
            where = ", ".join(str(i) for i in sorted(set(long_lines)))
            notes.append(f"Panel(s) {where} have long dialogue for this text size; the layout shrinks the lettering a little to fit.")
        shots = [(p.shot or "").lower() for p in panels]
        run = 1
        for i in range(1, len(shots)):
            run = run + 1 if shots[i] and shots[i] == shots[i - 1] else 1
            if run >= 3:
                notes.append(f"Panels {i - 1}-{i + 1} use the same shot; consider asking the assistant for more camera variety.")
                break
        if len(panels) >= 3 and not any(p.dialogue or p.caption for p in panels[-1:]):
            notes.append("The last panel has no text; fine for a silent beat, or ask the assistant to add a punchline.")
    except Exception:
        pass
    return notes
