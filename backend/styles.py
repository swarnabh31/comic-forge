"""Art style registry.

Before, an "art style" was one loose phrase pasted at the END of every image prompt, and the
same generic negative prompt was used for everything. That is why styles drifted between panels
and why some styles couldn't work at all (the old negative prompt banned "3D render" and
"photo" for every style, whatever you picked).

A style is now a small, complete spec:

  prompt     the positive style block. It goes FIRST in the image prompt, where the model weighs it
             most (and where SDXL's 77-token window can't cut it off).
  lock       a short reminder appended after the scene ("same linework and palette in every panel").
  negative   extra negative terms for the SDXL path (Qwen has no negative prompt).
  theme      how the LETTERING looks in this style: caption colour, bubble outline, paper, caps.
             Lettering is part of the style; a noir comic and a storybook shouldn't share bubbles.

`resolve()` accepts a style id, a style label, one of the five legacy phrases the old UI sent, or
any free text (a custom style), so old saved comics and chat edits like "make it watercolor" keep
working.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtStyle:
    id: str
    label: str
    group: str
    description: str
    prompt: str
    lock: str = "consistent linework, palette and shading in every panel"
    negative: str = ""
    three_d: bool = False          # a 3D look is the point of this style (so '3D render' isn't negated)
    mono: bool = False             # black-and-white style (colour is negated)
    theme: dict = field(default_factory=dict)

    @property
    def short(self) -> str:
        """A few words naming the style, used in 'redraw in X' instructions."""
        return self.label.lower()


# Lettering themes. Keys: paper, ink, caption_bg, caption_fg, bubble_bg, bubble_fg, caps
_CLASSIC = dict(paper=(245, 242, 232), ink=(17, 17, 17), caption_bg=(255, 199, 0), caption_fg=(17, 17, 17),
                bubble_bg=(255, 255, 255), bubble_fg=(17, 17, 17), caps=False)
_CAPS = {**_CLASSIC, "caps": True}
_MANGA = dict(paper=(255, 255, 255), ink=(10, 10, 10), caption_bg=(255, 255, 255), caption_fg=(10, 10, 10),
              bubble_bg=(255, 255, 255), bubble_fg=(10, 10, 10), caps=False)
_NOIR = dict(paper=(222, 220, 214), ink=(8, 8, 8), caption_bg=(236, 233, 224), caption_fg=(12, 12, 12),
             bubble_bg=(250, 249, 245), bubble_fg=(12, 12, 12), caps=True)
_SOFT = dict(paper=(250, 244, 230), ink=(74, 52, 40), caption_bg=(255, 240, 200), caption_fg=(74, 52, 40),
             bubble_bg=(255, 252, 244), bubble_fg=(74, 52, 40), caps=False)
_PASTEL = dict(paper=(255, 248, 250), ink=(92, 60, 92), caption_bg=(255, 226, 238), caption_fg=(70, 40, 70),
               bubble_bg=(255, 255, 255), bubble_fg=(70, 40, 70), caps=False)
_NEON = dict(paper=(18, 16, 30), ink=(8, 8, 16), caption_bg=(0, 229, 255), caption_fg=(8, 8, 16),
             bubble_bg=(250, 250, 255), bubble_fg=(12, 12, 24), caps=True)
_POP = dict(paper=(255, 244, 200), ink=(10, 10, 10), caption_bg=(255, 224, 0), caption_fg=(10, 10, 10),
            bubble_bg=(255, 255, 255), bubble_fg=(10, 10, 10), caps=True)
_COOL = dict(paper=(240, 244, 248), ink=(25, 35, 55), caption_bg=(255, 255, 255), caption_fg=(25, 35, 55),
             bubble_bg=(255, 255, 255), bubble_fg=(25, 35, 55), caps=False)

_STYLES: list[ArtStyle] = [
    # ------------------------------------------------------------------ Western
    ArtStyle(
        "classic_superhero", "Classic superhero", "Western comics",
        "Bold ink outlines, halftone shading, saturated primaries.",
        "classic American superhero comic book art, bold black ink outlines with varied line weight, "
        "halftone dot shading, saturated primary colors, dynamic heroic poses, flat cel coloring",
        lock="same inking style, halftone shading and color palette in every panel",
        theme=_CAPS),
    ArtStyle(
        "modern_american", "Modern American comic", "Western comics",
        "Polished digital inks with painterly color and cinematic lighting.",
        "modern American comic book art, clean digital inks, painterly digital coloring, cinematic lighting "
        "and rim light, rich color grading, detailed anatomy",
        lock="same digital inking, rendering and color grading in every panel",
        theme=_CLASSIC),
    ArtStyle(
        "retro_silver_age", "Retro silver-age", "Western comics",
        "1960s printing look: Ben-Day dots, limited CMYK palette, off-white paper.",
        "vintage 1960s silver age comic book art, visible Ben-Day dot printing, limited CMYK color palette, "
        "thick confident outlines, slightly faded newsprint colors, bright flat colors",
        lock="same Ben-Day dots, limited palette and thick outlines in every panel",
        theme={**_CAPS, "paper": (246, 238, 214)}),
    ArtStyle(
        "pop_art", "Pop art", "Western comics",
        "Giant halftone dots, bold primary colors, thick black contour lines.",
        "pop art comic illustration, large halftone dots, bold primary colors red yellow blue, thick black "
        "contour lines, flat graphic color blocks, high contrast",
        lock="same halftone dots, primary palette and thick contours in every panel",
        theme=_POP),
    ArtStyle(
        "noir", "Noir", "Western comics",
        "Black ink, heavy shadows, high contrast, near-monochrome.",
        "noir comic book art, heavy black ink shadows, extreme high contrast chiaroscuro, near-monochrome "
        "palette of black white and grey with a single muted accent color, hatching, moody lighting",
        lock="same heavy black ink shadows and restrained palette in every panel",
        theme=_NOIR),
    ArtStyle(
        "newspaper_strip", "Newspaper strip", "Western comics",
        "Simple, readable shapes and flat colors, like a Sunday funnies strip.",
        "newspaper comic strip art, simple rounded shapes, clean uniform ink lines, flat bright colors with no "
        "gradients, minimal backgrounds, highly readable expressive faces",
        lock="same simple line weight and flat colors in every panel",
        theme=_CLASSIC),
    ArtStyle(
        "ligne_claire", "Clear line (European)", "Western comics",
        "Uniform clean outlines, flat color, richly detailed backgrounds.",
        "European clear line comic art (ligne claire), uniform thin dark outlines of constant weight, flat "
        "colors with no gradients and no hatching, meticulously detailed backgrounds, simple expressive characters",
        lock="same uniform line weight and flat color in every panel",
        theme=_COOL),
    ArtStyle(
        "indie_graphic_novel", "Indie graphic novel", "Western comics",
        "Loose expressive ink with muted watercolor-wash color.",
        "indie graphic novel illustration, loose expressive ink linework, muted desaturated watercolor wash "
        "coloring, textured paper, quiet naturalistic mood, slightly imperfect hand-drawn lines",
        lock="same loose ink line and muted wash palette in every panel",
        theme=_SOFT),
    # ------------------------------------------------------------------ Manga & anime
    ArtStyle(
        "manga_bw", "Manga (black & white)", "Manga & anime",
        "Screentone shading, expressive linework, speed lines, no color.",
        "black and white Japanese manga page art, crisp G-pen ink linework, screentone shading, speed lines and "
        "emotion lines, expressive eyes, dramatic framing, pure black and white with no color",
        lock="same crisp ink line and screentone shading in every panel, black and white only",
        negative="color, colorful, painting, gradients",
        mono=True, theme=_MANGA),
    ArtStyle(
        "shonen_action", "Anime action (color)", "Manga & anime",
        "Vivid cel-shaded anime with dynamic angles and energetic lighting.",
        "color anime key visual style, sharp clean cel shading, vivid saturated colors, dynamic action angles, "
        "glowing highlights, detailed expressive eyes, energetic lighting",
        lock="same cel shading, line quality and vivid palette in every panel",
        theme=_CLASSIC),
    ArtStyle(
        "shojo", "Shojo romance", "Manga & anime",
        "Soft pastel anime with sparkly eyes, delicate lines and flowers.",
        "shojo manga style in soft color, delicate thin linework, large sparkling eyes, pastel palette, soft glow "
        "and bokeh sparkles, decorative flowers and screentone sparkles, airy lighting",
        lock="same delicate linework and soft pastel palette in every panel",
        theme=_PASTEL),
    ArtStyle(
        "chibi_cute", "Chibi / cute", "Manga & anime",
        "Super-deformed big-head characters, bright and playful.",
        "cute chibi super-deformed anime style, characters with oversized heads and tiny bodies, simple dot eyes, "
        "thick soft outlines, bright candy colors, playful emotive effects like sweat drops and sparkles",
        lock="same chibi proportions, soft outlines and candy palette in every panel",
        theme=_PASTEL),
    ArtStyle(
        "webtoon", "Webtoon", "Manga & anime",
        "Korean-webtoon look: clean lines, soft cel shading, glossy color.",
        "Korean webtoon style digital art, clean thin lineart, soft cel shading with gentle gradients, glossy "
        "luminous colors, attractive stylized proportions, polished lighting",
        lock="same clean lineart, soft shading and glossy palette in every panel",
        theme=_COOL),
    # ------------------------------------------------------------------ Cartoon
    ArtStyle(
        "cartoon_modern", "Modern cartoon", "Cartoon",
        "Thick uniform outlines, flat bright colors, bouncy shapes.",
        "modern western TV cartoon style, thick uniform outlines, flat bright colors, simple geometric shapes, "
        "exaggerated squash and stretch, expressive faces, clean vector look",
        lock="same outline weight, flat colors and shape language in every panel",
        theme=_CLASSIC),
    ArtStyle(
        "vintage_cartoon", "Vintage cartoon", "Cartoon",
        "1930s-50s animation: rubber-hose limbs, muted warm colors, grain.",
        "vintage 1940s animated cartoon style, rubber hose limbs, round pie-cut eyes, warm muted colors, "
        "painted backgrounds, subtle film grain and aged paper texture",
        lock="same rubber-hose character style and warm muted palette in every panel",
        theme=_SOFT),
    ArtStyle(
        "flat_vector", "Flat vector", "Cartoon",
        "Minimal geometric shapes, flat color, no outlines.",
        "flat vector illustration, minimal geometric shapes, flat solid colors with no outlines and no "
        "gradients, limited harmonious palette, clean modern graphic design",
        lock="same flat shapes and limited palette in every panel",
        theme=_COOL),
    ArtStyle(
        "three_d_animated", "3D animated film", "Cartoon",
        "Soft 3D character render with warm cinematic lighting.",
        "stylized 3D animated feature film look, soft rounded character models, expressive faces, subsurface "
        "skin scattering, warm cinematic lighting, shallow depth of field, rich textured sets",
        lock="same 3D character modelling, materials and lighting in every panel",
        three_d=True, theme=_CLASSIC),
    # ------------------------------------------------------------------ Painterly & illustrated
    ArtStyle(
        "storybook", "Storybook", "Illustrated",
        "Whimsical children's-book illustration, soft textures.",
        "whimsical children's storybook illustration, soft gouache and colored pencil textures, gentle rounded "
        "shapes, warm friendly palette, dreamy lighting",
        lock="same soft gouache texture and warm palette in every panel",
        theme=_SOFT),
    ArtStyle(
        "watercolor", "Watercolor", "Illustrated",
        "Loose ink lines with translucent watercolor washes and paper grain.",
        "watercolor illustration, loose ink pen linework, translucent watercolor washes with soft bleeding "
        "edges, visible paper grain, airy light palette",
        lock="same ink line and watercolor wash treatment in every panel",
        theme=_SOFT),
    ArtStyle(
        "pencil_sketch", "Pencil sketch", "Illustrated",
        "Graphite sketchbook drawing with cross-hatching.",
        "hand-drawn graphite pencil sketch, cross-hatching and soft shading, visible construction lines, "
        "sketchbook paper texture, monochrome grey tones",
        lock="same pencil line quality and hatching in every panel",
        negative="color, colorful, painting",
        mono=True, theme={**_SOFT, "paper": (240, 238, 232), "ink": (40, 40, 44), "bubble_fg": (30, 30, 34),
                          "caption_fg": (30, 30, 34), "caption_bg": (250, 249, 245)}),
    ArtStyle(
        "cyberpunk_neon", "Cyberpunk neon", "Illustrated",
        "Rain-slick futurist cities, neon magenta and cyan glow.",
        "cyberpunk comic illustration, neon magenta and cyan lighting, dark rainy futuristic city, glowing "
        "signage reflections, bold ink outlines, high contrast with vivid glow",
        lock="same neon palette, ink outlines and glow treatment in every panel",
        theme=_NEON),
]

STYLES: dict[str, ArtStyle] = {s.id: s for s in _STYLES}
DEFAULT_STYLE_ID = "classic_superhero"

# The five phrases the old UI <select> sent. Saved comics and API clients may still use them.
LEGACY: dict[str, str] = {
    "classic superhero comic, bold ink outlines, halftone shading": "classic_superhero",
    "classic superhero comic book, bold ink outlines, halftone shading": "classic_superhero",
    "manga style, screentone shading, expressive linework": "manga_bw",
    "newspaper comic strip, simple flat colors": "newspaper_strip",
    "noir comic, heavy shadows, high contrast ink": "noir",
    "whimsical children's storybook illustration": "storybook",
}

# Keywords for loose chat requests ("switch to anime", "make it watercolour").
_ALIASES: dict[str, str] = {
    "superhero": "classic_superhero", "marvel": "classic_superhero", "dc": "classic_superhero",
    "manga": "manga_bw", "anime": "shonen_action", "shonen": "shonen_action", "shojo": "shojo",
    "chibi": "chibi_cute", "webtoon": "webtoon", "noir": "noir", "film noir": "noir",
    "newspaper": "newspaper_strip", "funnies": "newspaper_strip", "strip": "newspaper_strip",
    "storybook": "storybook", "children's book": "storybook", "watercolor": "watercolor",
    "watercolour": "watercolor", "pop art": "pop_art", "silver age": "retro_silver_age",
    "retro": "retro_silver_age", "ligne claire": "ligne_claire", "clear line": "ligne_claire",
    "graphic novel": "indie_graphic_novel", "cartoon": "cartoon_modern", "vintage cartoon": "vintage_cartoon",
    "vector": "flat_vector", "3d": "three_d_animated", "pixar": "three_d_animated", "pencil": "pencil_sketch",
    "sketch": "pencil_sketch", "cyberpunk": "cyberpunk_neon", "neon": "cyberpunk_neon",
}

BASE_NEGATIVE = (
    "blurry, deformed, extra limbs, extra fingers, bad anatomy, watermark, signature, text, words, letters, "
    "speech bubble, caption box, low quality, collage, grid layout, multiple views, multiple panels, comic strip, "
    "product page, spec sheet, infographic, diagram, catalog, labels, callout boxes, gibberish text, "
    "UI screenshot, marketing page, photograph, photorealistic, style inconsistency, mixed art styles"
)


def get(style_id: str) -> ArtStyle | None:
    return STYLES.get((style_id or "").strip())


def resolve(value: "str | ArtStyle | None") -> ArtStyle:
    """Any style reference -> an ArtStyle. Unknown free text becomes a custom style built from it."""
    if isinstance(value, ArtStyle):
        return value
    v = (value or "").strip()
    if not v:
        return STYLES[DEFAULT_STYLE_ID]
    if v in STYLES:
        return STYLES[v]
    low = v.lower()
    if low in LEGACY:
        return STYLES[LEGACY[low]]
    for s in _STYLES:
        if low == s.label.lower():
            return s
    # A short loose request ("anime", "make it watercolour"): match on a keyword. Long phrases are
    # treated as a deliberate custom style instead.
    if len(low) <= 40:
        for key in sorted(_ALIASES, key=len, reverse=True):
            if key in low:
                return STYLES[_ALIASES[key]]
    return ArtStyle("custom", v[:40], "Custom", v, v[:240], theme=_CLASSIC)


def canonical(value: "str | ArtStyle | None") -> str:
    """The string to store in a comic's state: the style id when known, else the custom phrase."""
    s = resolve(value)
    return s.id if s.id != "custom" else (s.description or "").strip()[:200]


def negative_prompt(style: ArtStyle) -> str:
    parts = [BASE_NEGATIVE]
    if not style.three_d:
        parts.append("3D render, CGI")
    if style.negative:
        parts.append(style.negative)
    return ", ".join(parts)


def catalogue() -> list[dict]:
    """What the UI needs to build its picker, grouped and in a stable order."""
    return [{"id": s.id, "label": s.label, "group": s.group, "description": s.description} for s in _STYLES]


def ids_for_prompt() -> str:
    return ", ".join(s.id for s in _STYLES)
