"""Lettering and page layout.

Turns generated panel art plus the script's captions and dialogue into one finished comic page.

Why this is more than "draw some rectangles":

* TEXT SIZE is a first-class setting (small / medium / large). The font size is a fraction of the
  panel width, so it looks the same whatever resolution the image model produced.
* BUBBLES ADAPT TO THE TEXT. Text is wrapped into balanced lines (no orphan words), the bubble is
  sized to the wrapped block, and speech bubbles, thought clouds and caption boxes each have their own
  shape. Everything is drawn at 2x and downsampled, so edges are smooth.
* BUBBLES AVOID THE ART. An edge-energy map of the panel finds calm regions (sky, walls, empty
  floor). A search places each bubble where it hides the least detail, outside the speaker's zone
  (the side of the frame the script says they stand on), in reading order, with no overlaps.
* TAILS POINT AT THE SPEAKER, with a length that scales with the lettering.
* NOTHING IS ALLOWED TO LOOK WRONG. If the chosen size cannot be placed cleanly in a panel (a very
  long line at "large"), the lettering for THAT panel is shrunk in small steps until it fits, and the
  caller gets a note saying so, instead of text running off the edge or covering a face.
* PAGES NEVER HAVE EMPTY CELLS. Panel counts that don't fill a grid (5, 7, ...) get a row template
  with the short rows centred, rather than blank squares.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from schemas import ComicScript

ASSETS = Path(__file__).parent / "assets" / "fonts"
DISPLAY_FONT = ASSETS / "Bangers-Regular.ttf"
LETTER_FONT = ASSETS / "ComicNeue-Bold.ttf"

SS = 2                                   # supersampling factor for the lettering layer
# Font size as a fraction of PANEL WIDTH (1024px panel: 33 / 44 / 57 px).
TEXT_SIZES = {"small": 0.032, "medium": 0.043, "large": 0.056}
DEFAULT_TEXT_SIZE = "medium"
MIN_SHRINK = 0.70                        # never shrink lettering below 70% of the chosen size on purpose
MAX_BALLOONS = 3


# ----------------------------------------------------------------------------- theme


@dataclass
class Theme:
    paper: tuple = (245, 242, 232)
    ink: tuple = (17, 17, 17)
    caption_bg: tuple = (255, 199, 0)
    caption_fg: tuple = (17, 17, 17)
    bubble_bg: tuple = (255, 255, 255)
    bubble_fg: tuple = (17, 17, 17)
    caps: bool = False


def theme_for(style) -> Theme:
    """Lettering theme from an ArtStyle (or None for the classic look)."""
    t = Theme()
    for k, v in (getattr(style, "theme", None) or {}).items():
        if hasattr(t, k):
            setattr(t, k, v)
    return t


# ----------------------------------------------------------------------------- fonts & text


@lru_cache(maxsize=256)
def _font_cached(display: bool, size: int) -> ImageFont.FreeTypeFont:
    for path in ((DISPLAY_FONT,) if display else (LETTER_FONT, DISPLAY_FONT)):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _font(size: float, display: bool = False) -> ImageFont.FreeTypeFont:
    return _font_cached(display, max(8, int(round(size))))


_MEASURE = ImageDraw.Draw(Image.new("L", (1, 1)))


def _tw(text: str, font) -> float:
    return _MEASURE.textlength(text, font=font)


def wrap_balanced(text: str, font, max_w: float) -> list[str]:
    """Greedy wrap, then tighten the width to the narrowest that keeps the same line count, so lines
    are even and the last line is never a lone orphan word."""
    words = text.split()
    if not words:
        return []

    def greedy(width: float) -> list[str]:
        lines, line = [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if not line or _tw(trial, font) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
        return lines

    base = greedy(max_w)
    if len(base) == 1:
        return base
    lo, hi = max(_tw(w, font) for w in words), max_w
    if lo >= hi:
        return base
    for _ in range(14):
        mid = (lo + hi) / 2
        if len(greedy(mid)) <= len(base):
            hi = mid
        else:
            lo = mid
    return greedy(hi)


# ----------------------------------------------------------------------------- panel analysis


class EdgeMap:
    """Coarse 'how much is going on here' map of a panel, with O(1) rectangle averages.

    Bubbles should cover calm regions (sky, a wall) and not faces, hands or props. Detail shows up as edge
    energy, so we sum it with a summed-area table and query any rectangle cheaply."""

    G = 48

    def __init__(self, img: Image.Image):
        g = img.convert("L")
        g.thumbnail((256, 256))
        e = g.filter(ImageFilter.FIND_EDGES).resize((self.G, self.G), Image.BOX)
        vals = list(e.getdata())
        G = self.G
        S = [[0.0] * (G + 1) for _ in range(G + 1)]
        for y in range(G):
            row = 0.0
            for x in range(G):
                row += vals[y * G + x]
                S[y + 1][x + 1] = S[y][x + 1] + row
        self.S = S

    def mean(self, fx0: float, fy0: float, fx1: float, fy1: float) -> float:
        """Average edge energy in a rectangle given as FRACTIONS of the panel, normalised to ~0..1."""
        G = self.G
        x0 = max(0, min(G - 1, int(fx0 * G)))
        y0 = max(0, min(G - 1, int(fy0 * G)))
        x1 = max(x0 + 1, min(G, math.ceil(fx1 * G)))
        y1 = max(y0 + 1, min(G, math.ceil(fy1 * G)))
        S = self.S
        total = S[y1][x1] - S[y0][x1] - S[y1][x0] + S[y0][x0]
        return min(1.0, total / ((x1 - x0) * (y1 - y0)) / 28.0)


# ----------------------------------------------------------------------------- measuring


@dataclass
class Balloon:
    kind: str                      # "speech" | "thought"
    lines: list[str]
    font: ImageFont.FreeTypeFont
    font_px: float
    w: float
    h: float
    line_h: float
    # filled in by placement
    x0: float = 0.0
    y0: float = 0.0


def _measure_balloon(text: str, kind: str, font_px: float, panel_w: float, caps: bool) -> Balloon:
    font = _font(font_px)
    text = text.upper() if caps else text
    thought = kind == "thought"
    max_w = panel_w * (0.52 if thought else 0.60)
    lines = wrap_balanced(text, font, max_w)
    line_h = font_px * 1.16
    tw = max((_tw(t, font) for t in lines), default=0)
    th = line_h * len(lines)
    if thought:     # a cloud is an ellipse: the text block must fit inside it
        w, h = tw * 1.30 + font_px * 1.3, th * 1.38 + font_px * 1.0
    else:
        w, h = tw + font_px * 1.25, th + font_px * 0.9
    return Balloon(kind, lines, font, font_px, w, h, line_h)


@dataclass
class _Caption:
    lines: list[str]
    font: ImageFont.FreeTypeFont
    w: float
    h: float
    line_h: float
    pad: float


def _measure_caption(text: str, font_px: float, panel_w: float, caps: bool) -> _Caption:
    px = font_px * 0.92
    font = _font(px)
    pad = px * 0.55
    text = text.upper() if caps else text
    lines = wrap_balanced(text, font, panel_w * 0.90 - 2 * pad)
    line_h = px * 1.18
    tw = max((_tw(t, font) for t in lines), default=0)
    return _Caption(lines, font, tw + 2 * pad, line_h * len(lines) + 2 * pad * 0.8, line_h, pad)


# ----------------------------------------------------------------------------- placement


ANCHOR_X = {"left": 0.27, "center": 0.50, "right": 0.73}
ANCHOR_Y = 0.60                  # where the speaker's mouth usually is, as a fraction of panel height
BUBBLE_ZONE_BOTTOM = 0.70        # bubbles must end above this line, which leaves the lower art free


def _overlap(a, b, gap=0.0) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _inter_ratio(a, zone) -> float:
    ix = max(0.0, min(a[2], zone[2]) - max(a[0], zone[0]))
    iy = max(0.0, min(a[3], zone[3]) - max(a[1], zone[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def _place_balloons(balloons: list[Balloon], W: float, H: float, top_y: float, occupied: list, emap: EdgeMap,
                    placement: str, margin: float, zone_bottom: float = BUBBLE_ZONE_BOTTOM) -> list[tuple] | None:
    """Pick a position for each balloon, in reading order. Returns [(x0, y0)] or None if it can't fit."""
    ax = ANCHOR_X.get(placement, 0.5) * W
    ay = ANCHOR_Y * H
    zone = (ax - 0.17 * W, 0.36 * H, ax + 0.17 * W, H)       # where the speaker is: keep bubbles off them
    gap = margin * 0.8
    step_x, step_y = W / 36, H / 36
    taken = list(occupied)
    out: list[tuple] = []
    prev = None
    for b in balloons:
        best, best_cost = None, 1e9
        y = top_y
        y_max = zone_bottom * H - b.h
        while y <= y_max + 1e-6:
            x = margin
            while x <= W - b.w - margin + 1e-6:
                rect = (x, y, x + b.w, y + b.h)
                if not any(_overlap(rect, t, gap) for t in taken) and _reading_order_ok(rect, prev):
                    detail = emap.mean(rect[0] / W, rect[1] / H, rect[2] / W, rect[3] / H)
                    cx = (rect[0] + rect[2]) / 2
                    dist = math.hypot(cx - ax, rect[3] - ay) / H
                    cost = (3.2 * detail + 2.6 * _inter_ratio(rect, zone) + 1.1 * (y - top_y) / H + 0.9 * dist)
                    if prev is not None:
                        cost += 0.5 * max(0.0, 0.30 - abs(cx - (prev[0] + prev[2]) / 2) / W)   # stagger
                    if cost < best_cost:
                        best, best_cost = rect, cost
                x += step_x
            y += step_y
        if best is None:
            return None
        taken.append(_with_tail_room(best, b.font_px))     # tails hang below the bubble: keep that room free
        out.append((best[0], best[1]))
        prev = best
    return out


def _reading_order_ok(rect, prev) -> bool:
    """A later balloon must read after the earlier one: below it, or to its right on the same band."""
    if prev is None:
        return True
    below = rect[1] >= prev[1] + (prev[3] - prev[1]) * 0.55
    right_of = rect[0] >= prev[2] and rect[1] >= prev[1] - (prev[3] - prev[1]) * 0.15
    return below or right_of


def _with_tail_room(rect, font_px):
    return (rect[0], rect[1], rect[2], rect[3] + font_px * 1.0)


# ----------------------------------------------------------------------------- drawing


def _bezier(p0, c, p1, n=7):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]) for t in [i / n for i in range(n + 1)]]


def _tail_geometry(rect, ax, ay, font_px, r):
    """(base point, outward normal, tangent, tip, half-width) for a speech tail from `rect` toward (ax, ay)."""
    x0, y0, x1, y1 = rect
    half = font_px * 0.46
    length_cap = font_px * 2.6
    if ay > y1:                                                       # speaker below: tail off the bottom edge
        bx = min(max(ax, x0 + r + half), x1 - r - half)
        n, t = (0, 1), (1, 0)
        base = (bx, y1)
        dist = ay - y1
    elif ay < y0:                                                     # speaker above (rare): off the top edge
        bx = min(max(ax, x0 + r + half), x1 - r - half)
        n, t = (0, -1), (1, 0)
        base = (bx, y0)
        dist = y0 - ay
    elif ax < x0:                                                     # speaker to the left
        by = min(max(ay, y0 + r + half), y1 - r - half)
        n, t = (-1, 0), (0, 1)
        base = (x0, by)
        dist = x0 - ax
    else:
        by = min(max(ay, y0 + r + half), y1 - r - half)
        n, t = (1, 0), (0, 1)
        base = (x1, by)
        dist = ax - x1
    length = min(length_cap, max(font_px * 1.0, dist * 0.6))
    # lean the tip toward the anchor along the edge, without letting it flop sideways
    lean = (ax - base[0]) if t == (1, 0) else (ay - base[1])
    lean = max(-length * 0.8, min(length * 0.8, lean * 0.45))
    tip = (base[0] + n[0] * length + t[0] * lean, base[1] + n[1] * length + t[1] * lean)
    return base, n, t, tip, half


def _speech_shape(d: ImageDraw.ImageDraw, rect, tail, r, ol, theme: Theme):
    x0, y0, x1, y1 = rect
    outer_pts = fill_pts = None
    if tail:
        base, n, t, tip, half = tail
        bl = (base[0] - t[0] * half, base[1] - t[1] * half)
        br = (base[0] + t[0] * half, base[1] + t[1] * half)
        bend = half * 1.1
        left = _bezier(bl, ((bl[0] + tip[0]) / 2 + t[0] * bend, (bl[1] + tip[1]) / 2 + t[1] * bend), tip)
        right = _bezier(tip, ((br[0] + tip[0]) / 2 + t[0] * bend * 0.4, (br[1] + tip[1]) / 2 + t[1] * bend * 0.4), br)
        outer_pts = left + right[1:]
        # push the base inside the bubble so the tail fills over the bubble's own outline (no seam)
        inset = ol * 2.2
        bl_in = (bl[0] - n[0] * inset, bl[1] - n[1] * inset)
        br_in = (br[0] - n[0] * inset, br[1] - n[1] * inset)
        fill_pts = [bl_in] + outer_pts + [br_in]
    # ink silhouette first (bubble + tail grown by the outline width) ...
    d.rounded_rectangle((x0 - ol, y0 - ol, x1 + ol, y1 + ol), radius=r + ol, fill=theme.ink)
    if outer_pts:
        d.polygon(outer_pts, fill=theme.ink)
        d.line(outer_pts + [outer_pts[0]], fill=theme.ink, width=int(ol * 2), joint="curve")
    # ... then the paper-coloured fill on top
    d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=theme.bubble_bg)
    if fill_pts:
        d.polygon(fill_pts, fill=theme.bubble_bg)


def _thought_shape(d: ImageDraw.ImageDraw, rect, ax, ay, font_px, ol, theme: Theme):
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rr = font_px * 0.5
    a, b = (x1 - x0) / 2 - rr * 0.9, (y1 - y0) / 2 - rr * 0.9
    n = max(10, int(2 * math.pi * math.sqrt((a * a + b * b) / 2) / (rr * 1.45)))
    bumps = [(cx + a * math.cos(2 * math.pi * i / n), cy + b * math.sin(2 * math.pi * i / n)) for i in range(n)]
    # trail of ever-smaller circles toward the speaker
    dx, dy = ax - cx, ay - cy
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    k = 1.0 / math.sqrt((ux / (a + rr)) ** 2 + (uy / (b + rr)) ** 2)     # where the ray leaves the cloud
    sx, sy = cx + ux * k, cy + uy * k
    gap = min(font_px * 2.3, max(font_px * 1.3, (norm - k) * 0.6))
    trail = []
    for i, rad in enumerate((0.34, 0.24, 0.15)):
        f = 0.28 + 0.72 * ((i + 1) / 3.0) * 0.9
        trail.append((sx + ux * gap * f, sy + uy * gap * f, font_px * rad))
    for (px, py) in bumps:
        d.ellipse((px - rr - ol, py - rr - ol, px + rr + ol, py + rr + ol), fill=theme.ink)
    d.ellipse((cx - a - ol, cy - b - ol, cx + a + ol, cy + b + ol), fill=theme.ink)
    for (px, py, rad) in trail:
        d.ellipse((px - rad - ol, py - rad - ol, px + rad + ol, py + rad + ol), fill=theme.ink)
    for (px, py) in bumps:
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=theme.bubble_bg)
    d.ellipse((cx - a, cy - b, cx + a, cy + b), fill=theme.bubble_bg)
    for (px, py, rad) in trail:
        d.ellipse((px - rad, py - rad, px + rad, py + rad), fill=theme.bubble_bg)


def _draw_balloon(d: ImageDraw.ImageDraw, b: Balloon, ax: float, ay: float, theme: Theme):
    rect = (b.x0, b.y0, b.x0 + b.w, b.y0 + b.h)
    ol = max(2.5 * SS / 2, b.font_px * 0.075)
    if b.kind == "thought":
        _thought_shape(d, rect, ax, ay, b.font_px, ol, theme)
    else:
        r = min(b.h / 2, b.font_px * 0.95)
        _speech_shape(d, rect, _tail_geometry(rect, ax, ay, b.font_px, r), r, ol, theme)
    cx = b.x0 + b.w / 2
    y = b.y0 + (b.h - b.line_h * len(b.lines)) / 2 + b.line_h / 2
    for line in b.lines:
        d.text((cx, y), line, font=b.font, fill=theme.bubble_fg, anchor="mm")
        y += b.line_h


def _draw_caption(d: ImageDraw.ImageDraw, c: _Caption, x0: float, y0: float, ol: float, theme: Theme):
    d.rectangle((x0, y0, x0 + c.w, y0 + c.h), fill=theme.caption_bg, outline=theme.ink, width=int(ol))
    y = y0 + c.h / 2 - c.line_h * len(c.lines) / 2 + c.line_h / 2
    for line in c.lines:
        d.text((x0 + c.pad, y), line, font=c.font, fill=theme.caption_fg, anchor="lm")
        y += c.line_h


# ----------------------------------------------------------------------------- one panel


@dataclass
class LetterResult:
    image: Image.Image
    scale: float = 1.0                       # 1.0 = rendered at exactly the chosen size
    notes: list[str] = field(default_factory=list)


def letter_panel(img: Image.Image, panel, theme: Theme, text_size: str = DEFAULT_TEXT_SIZE) -> LetterResult:
    """Stamp the panel's caption and dialogue onto its artwork."""
    caption = (panel.caption or "").strip()
    lines = [(d.text.strip(), d.type) for d in panel.dialogue[:MAX_BALLOONS] if d.text.strip()]
    if not caption and not lines:
        return LetterResult(img.convert("RGB"))

    w, h = img.size
    W, H = w * SS, h * SS
    base_px = TEXT_SIZES.get(text_size, TEXT_SIZES[DEFAULT_TEXT_SIZE]) * W
    emap = EdgeMap(img)
    placement = getattr(panel, "placement", "center") or "center"
    ax, ay = ANCHOR_X[placement] * W, ANCHOR_Y * H

    # Each failed attempt shrinks the lettering a little AND (from attempt 3) lets bubbles use more of the
    # panel, so crowded panels get a cost-aware placement instead of a blind stack.
    zone_steps = (BUBBLE_ZONE_BOTTOM, BUBBLE_ZONE_BOTTOM, BUBBLE_ZONE_BOTTOM, 0.80, 0.80, 0.90, 0.92)
    factor, notes, plan = 1.0, [], None
    for _attempt in range(7):
        px = base_px * factor
        margin = max(10.0 * SS, px * 0.45)
        ol = max(2.5 * SS / 2, px * 0.075)
        cap = _measure_caption(caption, px, W, theme.caps) if caption else None
        balloons = [_measure_balloon(t, k, px, W, theme.caps) for t, k in lines]
        occupied, top_y = [], margin
        if cap:
            cap_rect = (margin, margin, margin + cap.w, margin + cap.h)
            occupied.append(cap_rect)
            top_y = cap_rect[3] + margin * 0.8
        fits = cap is None or cap.h < 0.30 * H
        if fits and all(b.w <= W - 2 * margin for b in balloons):
            positions = _place_balloons(balloons, W, H, top_y, occupied, emap, placement, margin, zone_steps[_attempt]) if balloons else []
            if positions is not None:
                plan = (cap, balloons, positions, margin, ol)
                break
        factor *= 0.92

    if plan is None:                         # last resort: stack from the top and accept it
        px = base_px * MIN_SHRINK
        margin = max(10.0 * SS, px * 0.45)
        ol = max(2.5 * SS / 2, px * 0.075)
        cap = _measure_caption(caption, px, W, theme.caps) if caption else None
        balloons = [_measure_balloon(t, k, px, W, theme.caps) for t, k in lines]
        y = margin + (cap.h + margin if cap else 0)
        positions = []
        for b in balloons:
            x = min(max(margin, ANCHOR_X[placement] * W - b.w / 2), max(margin, W - b.w - margin))
            positions.append((x, min(y, max(margin, H - b.h - margin))))
            y += b.h + margin
        plan = (cap, balloons, positions, margin, ol)
        factor = MIN_SHRINK
        notes.append("text is very long for this panel; lettering was packed in at reduced size")

    cap, balloons, positions, margin, ol = plan
    if factor < 0.999 and not notes:
        notes.append(f"lettering shrunk to {round(factor * 100)}% of the chosen size to fit")

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    if cap:
        _draw_caption(d, cap, margin, margin, ol, theme)
    for b, (x, y) in zip(balloons, positions):
        b.x0, b.y0 = x, y
        _draw_balloon(d, b, ax, ay, theme)
    overlay = overlay.resize((w, h), Image.LANCZOS)
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return LetterResult(out, factor, notes)


# ----------------------------------------------------------------------------- page


def plan_rows(n: int, max_cols: int = 3) -> list[int]:
    """Panels per row so that no cell is ever left empty (short rows are centred by the caller)."""
    if n <= 0:
        return []
    if max_cols <= 2:
        rows = [2] * (n // 2) + ([1] if n % 2 else [])
        return rows or [1]
    templates = {1: [1], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2], 6: [3, 3], 7: [2, 3, 2], 8: [3, 3, 2]}
    if n in templates:
        return templates[n]
    rows, left = [], n
    while left > 0:
        take = min(3, left)
        rows.append(take)
        left -= take
    return rows


def compose_page_with_notes(script: ComicScript, panel_images: list[Image.Image], style=None,
                            text_size: str = DEFAULT_TEXT_SIZE) -> tuple[Image.Image, list[str]]:
    """Letter every panel, lay them on a page with a title banner. Returns (page, notes)."""
    if len(script.panels) != len(panel_images):
        raise ValueError(f"{len(script.panels)} panels in the script but {len(panel_images)} images")
    if not panel_images:
        raise ValueError("a comic needs at least one panel")
    theme = theme_for(style)
    pw0, ph0 = panel_images[0].size
    border = max(4, round(pw0 * 0.006))
    gutter = max(14, round(pw0 * 0.014))

    finished, notes = [], []
    for i, (panel, img) in enumerate(zip(script.panels, panel_images), start=1):
        if img.size != (pw0, ph0):
            img = img.resize((pw0, ph0), Image.LANCZOS)
        res = letter_panel(img, panel, theme, text_size)
        notes += [f"Panel {i}: {n}." for n in res.notes]
        framed = Image.new("RGB", (pw0 + border * 2, ph0 + border * 2), theme.ink)
        framed.paste(res.image, (border, border))
        finished.append(framed)

    pw, ph = finished[0].size
    # Large text reads better on a narrower page (the preview is scaled to the screen width).
    rows = plan_rows(len(finished), max_cols=2 if text_size == "large" else 3)
    cols = max(rows)
    page_w = cols * pw + gutter * (cols + 1)

    title_px = max(40, round(pw0 * 0.075))
    title_font = _font(title_px, display=True)
    while title_px > 24 and _tw(script.title, title_font) > page_w * 0.92:
        title_px -= 3
        title_font = _font(title_px, display=True)
    title_h = round(title_px * 1.9)

    page = Image.new("RGB", (page_w, title_h + len(rows) * ph + gutter * (len(rows) + 1)), theme.paper)
    draw = ImageDraw.Draw(page)
    draw.text((page_w / 2, title_h / 2 + gutter * 0.3), script.title, font=title_font, fill=theme.ink, anchor="mm")

    idx = 0
    for r, count in enumerate(rows):
        row_w = count * pw + gutter * (count - 1)
        x_start = (page_w - row_w) // 2
        y = title_h + gutter + r * (ph + gutter)
        for c in range(count):
            page.paste(finished[idx], (x_start + c * (pw + gutter), y))
            idx += 1
    return page, notes


def compose_page(script: ComicScript, panel_images: list[Image.Image], style=None,
                 text_size: str = DEFAULT_TEXT_SIZE) -> Image.Image:
    return compose_page_with_notes(script, panel_images, style, text_size)[0]
