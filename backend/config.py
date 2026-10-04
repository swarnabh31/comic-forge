"""All runtime configuration lives here and is driven by environment variables
so the app can be pointed at different local/remote model providers without
touching code. Copy .env.example to .env and edit it."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
CHARACTERS_DIR = STORAGE_DIR / "characters"
COMICS_DIR = STORAGE_DIR / "comics"
for d in (STORAGE_DIR, CHARACTERS_DIR, COMICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Language model (script writing) -----------------------------------
# "ollama" -> local models, listed live via `ollama list` for a picker dropdown
# "hf"     -> Hugging Face Inference API chat model, fixed by HF_LLM_MODEL
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# --- Panel artwork ---
# "qwen"        -> Qwen-Image-2.1 (default; strongest character identity)
# "local"       -> SDXL + IP-Adapter on your GPU. Character consistency depends on
#                  every character having a reference portrait (uploaded, or
#                  auto-generated from the script's cast description).
# "hf_inference" -> Hugging Face Inference API (no local GPU needed, no character-reference conditioning)
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "qwen")

SD_MODEL_ID = os.getenv("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_REPO = os.getenv("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_WEIGHT_SDXL = os.getenv("IP_ADAPTER_WEIGHT_SDXL", "ip-adapter_sdxl.bin")
DEVICE = os.getenv("DEVICE", "cuda")  # "cuda" | "cpu" | "mps"
# 40 steps: enough that the identity/style signals settle instead of being
# truncated mid-guess (this was a contributor to per-panel style drift). Lower
# here if you're on a smaller GPU and the panels take too long.
IMAGE_STEPS = int(os.getenv("IMAGE_STEPS", "40"))
PANEL_SIZE = int(os.getenv("PANEL_SIZE", "768"))

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

MAX_PANELS = int(os.getenv("MAX_PANELS", "8"))
MAX_CHARACTERS = int(os.getenv("MAX_CHARACTERS", "6"))


# --- Qwen-Image-2.1 provider (config.IMAGE_PROVIDER = "qwen") ---
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1")
QWEN_STEPS = int(os.getenv("QWEN_STEPS", "20"))
QWEN_PANEL_SIZE = int(os.getenv("QWEN_PANEL_SIZE", "1024"))

# --- Comic chat assistant ---------------------------------------------------
ATTACHMENTS_DIR = STORAGE_DIR / "attachments"   # deliberately NOT served over HTTP
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_ATTACHMENTS = int(os.getenv("MAX_ATTACHMENTS", "5"))            # files per chat message
MAX_ATTACHMENT_MB = int(os.getenv("MAX_ATTACHMENT_MB", "15"))       # per file
ATTACHMENT_TEXT_BUDGET = int(os.getenv("ATTACHMENT_TEXT_BUDGET", "12000"))  # chars of document text shown to the LLM
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "12"))     # past chat messages sent to the LLM
MAX_VERSIONS = int(os.getenv("MAX_VERSIONS", "30"))                 # undo depth per comic


# --- Comic-craft knowledge (backend/knowledge/comics_expert.md) ---------------
# "condensed" -> a distilled rule set (~900 tokens) injected into the script writer and the
#                chat assistant. Fast, and safe for small context windows. (default)
# "full"      -> the whole comics_expert.md is sent to the script writer (~6.5k tokens). Best
#                craft quality; needs OLLAMA_NUM_CTX >= 16384 and adds a few seconds of prompt
#                processing. The chat assistant always uses the condensed rules.
# "off"       -> no craft knowledge (the original behaviour).
COMICS_EXPERT = os.getenv("COMICS_EXPERT", "condensed").lower()

# --- LLM runtime ------------------------------------------------------------
# Ollama's default context window is only 2-4k tokens and it SILENTLY truncates the start of
# the prompt (i.e. the instructions). Always ask for a window big enough for the prompts.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "5m")          # how long Ollama keeps the model resident after a call
# Free the LLM's VRAM right before drawing starts, so the image model doesn't fight it for memory.
UNLOAD_LLM_BEFORE_DRAWING = os.getenv("UNLOAD_LLM_BEFORE_DRAWING", "1") == "1"

# --- Lettering / text size ---------------------------------------------------
DEFAULT_TEXT_SIZE = os.getenv("DEFAULT_TEXT_SIZE", "medium")   # small | medium | large

# --- Speed / quality ---------------------------------------------------------
# Presets scale the configured panel size and step count. "draft" is ~2.5x faster than
# "standard" and is meant for trying out a script; "high" spends more steps on detail.
QUALITY_PRESETS = {
    "draft":    {"size": 0.75, "steps": 0.6},
    "standard": {"size": 1.0,  "steps": 1.0},
    "high":     {"size": 1.0,  "steps": 1.4},
}
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "standard")
# Character portraits are only used as conditioning, so they don't need full resolution.
PORTRAIT_SIZE_SCALE = float(os.getenv("PORTRAIT_SIZE_SCALE", "0.75"))
# Reference images are shrunk to this before conditioning: a 4000px phone photo makes every panel slower.
REF_MAX_SIDE = int(os.getenv("REF_MAX_SIDE", "768"))
# Load the image model in the background while the LLM writes the script (hides the 20-30 s load).
PRELOAD_IMAGE_MODEL = os.getenv("PRELOAD_IMAGE_MODEL", "1") == "1"

# --- Art style consistency ---------------------------------------------------
# Re-draw every UPLOADED character once, in the chosen art style, and use that portrait for the
# panels. Without this the style of the uploaded picture (photo, anime, ...) leaks into the panels
# and the comic mixes several looks. Costs one small image per uploaded character.
STYLE_NORMALIZE_REFS = os.getenv("STYLE_NORMALIZE_REFS", "1") == "1"

# --- Continuity (same room, same props, sane perspective) ---------------------
# Each distinct setting is defined ONCE in the script (a "location bible"). Its text is placed in every panel
# that happens there. With the Qwen engine, one empty "plate" image of the location is also drawn and passed
# to every panel in it as a reference, which is what actually pins down layout and props.
LOCATION_PLATES = os.getenv("LOCATION_PLATES", "1") == "1"
MAX_LOCATIONS = int(os.getenv("MAX_LOCATIONS", "3"))        # fewer locations = better continuity
# Dutch tilt / bird's-eye / worm's-eye are where image models break perspective. Off by default.
ALLOW_EXTREME_ANGLES = os.getenv("ALLOW_EXTREME_ANGLES", "0") == "1"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))        # per character reference image
