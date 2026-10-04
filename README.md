# Comic Forge

Type one idea, get back a full multi-panel comic: script, artwork, captions, dialogue,
and a laid-out page you can preview and download. Upload reference photos or sketches
for your characters and the app keeps drawing them consistently across panels — and
even characters you don't upload get a consistent look.

---

## Built & tested on

This app was developed and verified end-to-end on this stack — if your machine
looks like the table below, expect it to work out of the box:

| Component | What was used |
|---|---|
| LLM (script writing + chat assistant) | **qwen3.8:27b** (17 GB, local via Ollama) |
| Image generation (default path) | **Qwen-Image-2.1** (~33 GB bf16, CPU-offloaded onto the GPU) |
| Image generation (lighter path) | **Stable Diffusion XL** base 1.0 + **IP-Adapter** |
| GPU | **NVIDIA RTX 5090, 32 GB VRAM** |
| OS / Python | Windows 11, Python 3.11.15 in a conda env |

The exact package versions that were verified working on this setup are in the
section [Dependencies (exact versions)](#dependencies-exact-versions) below.

---

## Table of contents

- [Built & tested on](#built--tested-on)
- [How it works](#how-it-works)
- [Requirements (hardware + software)](#requirements-hardware--software)
- [Dependencies (exact versions)](#dependencies-exact-versions)
- [Model downloads](#model-downloads)
- [Setup](#setup)
- [Run](#run)
- [Using it](#using-it)
- [Editing a comic with the chat assistant](#editing-a-comic-with-the-chat-assistant)
- [Switching image providers](#switching-image-providers)
- [Troubleshooting](#troubleshooting)

---

## How it works

1. **Script** — your idea + cast of character names is sent to an LLM (local Ollama
   model, or a Hugging Face hosted model). It returns a structured script: a title,
   a **cast list** (every recurring character or entity — including non-human ones,
   like a talking fridge — with a written appearance description), and N panels each
   with a scene description, an optional caption, and dialogue lines. A caption is
   never allowed to just repeat a dialogue line — the prompt forbids it, and a
   safety net in `llm.py` drops the caption if it slips through anyway. A separate
   post-processor enforces "one speaker per panel" so the model doesn't cram two
   speakers into one shot.
2. **Cast reference portraits** — anyone in the cast who wasn't given an uploaded
   photo gets exactly one canonical portrait generated from their appearance
   description, so even characters you never uploaded stay visually consistent
   across the whole comic, instead of being redrawn from scratch every panel.
3. **Artwork** — one of two image engines draws each panel (switchable via
   `IMAGE_PROVIDER` in `.env`):
   - **`qwen`** (now the default) — **Qwen-Image-2.1**, a 7B DiT that takes a
     reference image (or list of reference images) as native conditioning input.
     Identity transfer for *both* people and non-human objects (fridge, robot,
     car) is significantly stronger than IP-Adapter blending.
   - **`local`** — Stable Diffusion XL + IP-Adapter. Panels are conditioned on
     the cast portrait(s) via IP-Adapter.
   - A single seed is generated once per comic and reused (with a small per-panel
     offset) across every portrait and panel, so line weight and palette don't
     reset between panels.
4. **Layout** — panels are lettered and stitched into one page with Pillow: a comic font
   (Comic Neue), bubbles sized to the text and placed over the calmest part of each
   panel, tails pointing at the speaker, thought clouds, caption boxes, and a lettering
   theme that matches the art style. See [Text size](#text-size-in-bubbles).

---

## Requirements (hardware + software)

### Hardware

| Component | Minimum | Recommended (Qwen path) |
|---|---|---|
| GPU (NVIDIA) | RTX 3060 12 GB (SDXL path only, smaller panels) | **RTX 4090 / 5090 24–32 GB** |
| System RAM | 16 GB | 32 GB |
| Disk (free) | ~8 GB (SDXL + IP-Adapter) | **~35 GB** (Qwen weights + both models cached) |
| Bandwidth (one-time) | — | ~35 GB for first Qwen download |

> **Why Qwen needs more VRAM:** Qwen-Image-2.1 is ~33 GB in bf16. On a 32 GB
> card, `enable_model_cpu_offload()` (used by `qwen_gen.py`) shards weights
> between CPU RAM and GPU so it fits. On a 24 GB card, lower `QWEN_PANEL_SIZE`
> to 768 and `QWEN_STEPS` to 12.

### Software

- **Python 3.11** (3.10 or 3.12 also work, but 3.11 is what's been tested here)
- **NVIDIA driver** with CUDA 12.x or newer (the `aispace` conda env here uses
  cu130 wheels — both work; install whatever `pip` resolves for your torch)
- **Ollama** running locally (for script writing), or a Hugging Face account
  (`HF_TOKEN`) if you'd rather use a hosted LLM
- **Git** (only if cloning)
- **Optional:** a conda or venv manager to keep the deps isolated from system
  Python

---

## Dependencies (exact versions)

The following were **verified working end-to-end** on this setup (Windows 11,
RTX 5090, Python 3.11.15, `aispace` conda env):

| Package | Version | Notes |
|---|---|---|
| **fastapi** | 0.136.1 | API + job orchestration |
| **uvicorn** | 0.46.0 | ASGI server |
| **python-multipart** | 0.0.26 | File uploads |
| **pydantic** | 2.12.5 | Schemas |
| **python-dotenv** | 1.x | `.env` loading |
| **httpx** | ≥0.27 | HF Inference client |
| **Pillow** | 11.3.0 | Layout + image IO |
| **pymupdf** | ≥1.24 | Chat attachments: read PDFs, render SVGs |
| **python-docx** | ≥1.1 | Chat attachments: read Word (.docx) files |
| **torch** | **2.9.1+cu130** | ⚠️ See downgrade note below |
| **torchvision** | **0.24.1+cu130** | ⚠️ Required by Qwen-Image-2.1 (Qwen3-VL video processor). **Not** needed for SDXL path. |
| **diffusers** | **0.41.0.dev0** | Qwen-Image-2.1 pipeline lives here |
| **transformers** | **5.17.0** | Qwen tokenizer + Qwen3-VL processor |
| **accelerate** | 1.13.0 | Model offloading |
| **huggingface_hub** | 1.33.0 | `snapshot_download` |
| **safetensors** | 0.8.0 | Weight format |
| **einops** | 0.8.2 | Transformer rearrange ops |
| **sentencepiece** / **tokenizers** | 0.1.99 / 0.23.2 | Tokenizer backends |
| **numpy** | 2.1.3 | (Some ML packages pin <2.0 — see Troubleshooting) |

### ⚠️ Torch downgrade note (Windows + Qwen path)

The Qwen-Image-2.1 pipeline (as shipped in diffusers 0.41) loads a **Qwen3-VL
video processor** that requires `torchvision`. `torchvision` wheels are only
published in **matched pairs** with specific `torch` versions. If you install
`torchvision` via plain `pip install torchvision`, pip will pick a version
that needs a specific torch — and it will **downgrade `torch`** to the matching
pair. This happened here: torch went from **2.13 → 2.9.1** when torchvision
0.24.1 was added.

**This is expected and fine** — the app was re-verified to work on torch 2.9.1
+ torchvision 0.24.1 + diffusers 0.41 + transformers 5.17. If you're on a
different torch, either:

1. Keep the torch version and check whether `import torchvision` works
   against it (it will on most recent CUDA builds), **or**
2. Let pip resolve the pair — `pip install torchvision` — and accept the
   torch downgrade, **or**
3. Install a specific paired set:
   ```bash
   pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
   ```

If you only use the SDXL path, you can skip `torchvision` entirely — it isn't
imported by `imagegen.py`.

### Install in one shot

```bash
pip install -r requirements.txt
# if you want the Qwen path (the default in this project):
pip install torchvision   # or a specific paired version, see note above
python tools/prefetch_models.py   # one-time: pre-download + pre-warm the image model (see below)
```

---

## Model downloads

Models are cached in `~/.cache/huggingface/hub/` (or set `HF_HOME` to
redirect). After a one-time download, the app works fully offline. **You only
need the row for the `IMAGE_PROVIDER` you'll actually use** — the app never
downloads both:

| Model | Repo | Size | Used by |
|---|---|---|---|
| **Qwen-Image-2.1** | `Qwen/Qwen-Image-2.1` | **~33 GB** | `IMAGE_PROVIDER=qwen` (default) |
| **Stable Diffusion XL** | `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | `IMAGE_PROVIDER=local` |
| **IP-Adapter (SDXL)** | `h94/IP-Adapter` (weights only, `ip-adapter_sdxl.bin`) | ~1 GB | `IMAGE_PROVIDER=local` |

You don't need to download models by hand: the diffusers pipelines auto-pull
on first use (i.e. your **first comic request**). To move that one-time
download out of the first request — with a progress bar and resume-safety —
run the prefetch script described in
[Pre-warming the image models](#-pre-warming-the-image-models-do-this-once-before-your-first-comic)
in the [Setup](#setup) section. See
[Troubleshooting](#troubleshooting) for a resume-safe manual download command.

---

## Setup

```bash
conda create -n comicforge python=3.11 -y
conda activate comicforge
cd comic-forge
pip install -r requirements.txt
pip install torchvision          # for the Qwen path (default); see torch note above
cp .env.example .env            # edit to taste
```

Make sure Ollama is running and has at least one model pulled:

```bash
ollama serve                     # if not already running in another terminal
ollama pull qwen3.8:27b          # the model this was built & tested with
```

The app reads `ollama list` live and shows every local model in a dropdown, so
nothing is hardcoded. (Any instruction-tuned 7B+ model works, but `qwen3.8:27b`
is the one the script and chat assistant were verified against.)

### ⏬ Pre-warming the image models (do this once, before your first comic)

**Why this matters:** the app downloads its image model **lazily** — neither
`uvicorn` startup nor merely importing the app fetches anything. The **first
"Generate comic!" click** is the one that pays for the one-time download
(~33 GB for Qwen-Image-2.1) **plus** the one-time pipeline build (~20–30 s).
On a slow or flaky connection that first click looks like the app is frozen
when it is just downloading.

**Fix:** run the prefetch script once after `pip install` — it does exactly
that one-time work now, with progress bars you can watch, and it reads your
`.env` so it targets whichever `IMAGE_PROVIDER` you have active:

```bash
python tools/prefetch_models.py
```

What it does, step by step:

1. **Checks the local cache first (no download yet).** It looks for the model
   in `~/.cache/huggingface/hub/` — the standard Hugging Face cache. If you
   already downloaded it for another project, you'll see
   `✅ <model>: all files present locally — no download needed` and the script
   skips straight to step 3. **Nothing is ever re-downloaded if all files are
   already there.** If it prints `⚠️ <model>: not (fully) present`, only the
   missing files are fetched (incomplete files resume, they don't restart).
2. Loads the pipeline the same way the app does (`QwenImage21Pipeline.from_pretrained`
   for the qwen path, or `StableDiffusionXLPipeline.from_pretrained` + IP-Adapter
   for the local path). **This is the one that downloads the weights** on a
   fresh machine where the model isn't cached yet.
3. Generates one throwaway image, which forces every component (VAE, text
   encoder, transformer) to initialize once.
4. Prints a done message.

**Already have the model downloaded for another task?** Nothing to do — the
app and this script both use the standard shared cache
(`~/.cache/huggingface/hub/`), so any Qwen-Image-2.1 / SDXL / IP-Adapter you
already have on disk is picked up automatically, and there is no second copy.
Downloaded it into a *custom* folder instead? Point `HF_HOME` (or
`HF_HUB_CACHE`) at that folder in your `.env` and both the app and the
prefetch script will use it.

After this, the first real "Generate comic!" request **skips the download and
the pipeline build** and goes straight to drawing — no more fake-hang on the
first click.

The script uses the **same HF cache the app uses**, so it doesn't add any
duplicate disk usage, and it doesn't touch `storage/` or generated comics.

`--force` flag:
```bash
python tools/prefetch_models.py --force
```
Deletes the locally cached weights (after a confirmation prompt) and
re-downloads them cleanly — the fix for a corrupt or truncated cache (e.g.
the model "loads" but the first panel gives a `CUDA error` or garbage
output).

---

## Run

From the project's top-level folder (the one containing `backend/`, `frontend/`,
`requirements.txt`):

```bash
uvicorn main:app --reload --app-dir backend
```

Then open `http://localhost:8000`.

The first comic request will trigger a one-time pipeline load (~20–30s for
Qwen on an RTX 5090), then each panel takes ~30–60s at 20 steps / 1024 px.
Subsequent requests reuse the in-memory pipeline.

---

## Using it

1. **(Optional)** In the **Cast** panel, add each character by name and upload
   one or more reference photos/sketches of them. Anyone you don't upload still
   gets a consistent look — the app generates a reference portrait for them
   automatically from the script.
2. Type your idea — a story seed, a joke, a "what if" — into the prompt box.
3. Pick a local model, panel count, **art style** (22 to choose from, or your own wording),
   **text size** (Small / Medium / Large) and **speed / quality**, then hit **Generate comic!**
4. Watch the progress bar. Each panel appears as a thumbnail the moment it is drawn.
   Preview the finished page and download it as a PNG.
5. Text too small or too big? Use the **Text size** buttons under the finished comic: the
   page is re-lettered in about a second, with no redrawing, and it can be undone.

### Text size in bubbles

Small / Medium / Large set the lettering as a fraction of the panel width (3.2% / 4.3% / 5.6%),
so it looks the same at any resolution. The setting is applied in three places so nothing looks
out of place:

- **The script writer** gets a word budget for the size (14 / 12 / 9 words per bubble, fewer lines
  at Large), so text is written to fit rather than squeezed afterwards.
- **The artist** is told to keep the top of the frame (20% / 30% / 38%) calm and uncluttered,
  and where the speaker stands, so there is empty space for the bubbles.
- **The layout engine** wraps text into even lines, sizes each bubble to its text, finds a calm
  spot with an edge-detail map (off the speaker's face, in reading order, no overlaps),
  and aims the tail at the speaker. If a very long line can't fit cleanly, only that panel's
  lettering is shrunk in small steps (never below ~70% unless unavoidable) and the app tells
  you which panel.

### Art styles

`GET /api/styles` lists them; the picker groups them: **Western comics** (classic superhero, modern
American, retro silver-age, pop art, noir, newspaper strip, clear line, indie graphic novel),
**Manga & anime** (B&W manga, anime action, shojo, chibi, webtoon), **Cartoon** (modern, vintage,
flat vector, 3D animated) and **Illustrated** (storybook, watercolor, pencil sketch, cyberpunk neon),
plus **Custom style…** for your own wording. Add or tweak styles in `backend/styles.py`: one entry
holds the prompt block, a "keep it consistent" line, extra negative terms, and the lettering theme.

### Same room, same props, sane perspective

Each script defines its **locations** once (layout, fixed props and where they stand, lighting). That text is repeated
verbatim in every panel set there; the panels share one seed; and with the Qwen engine an empty reference picture of the
room (the "plate") is passed to each panel. Camera angles are kept to eye level (mild low/high allowed; tilted and
bird's-eye views are clamped because image models can't draw them reliably). Related `.env` settings:
`LOCATION_PLATES` (default 1), `MAX_LOCATIONS` (default 3), `ALLOW_EXTREME_ANGLES` (default 0).
If a room still drifts, say so in the chat: *"change the boardroom: podium on the left, screen behind it"* redraws every
panel in that room from the new description.

### Speed / quality

| Preset | Size | Steps | Use it for |
|---|---|---|---|
| Draft | 75% | 60% | Trying ideas: roughly 2.5x faster |
| Standard | 100% | 100% | Default |
| High | 100% | 140% | The keeper |

---

## Editing a comic with the chat assistant

When a comic finishes, a **Edit with chat** button appears in the bottom-left corner.
The assistant is aware of the comic you just made — its script, cast, art style and
every panel — and applies your requests to it:

- *"Make the dialogue in panel 3 shorter"*, *"give the caption a spooky tone"*
- *"Redraw panel 2 from above"*, *"give me another take on panel 4"*
- *"Add a plot-twist panel at the end"*, *"remove panel 5"*, *"swap panels 2 and 3"*
- *"Make Arjun older with a grey beard"* (redraws his portrait and every panel he's in)
- *"Switch to noir"* or *"make it watercolor"* (redraws everything in the new style; uploaded characters are re-styled too)
- *"Make the text bigger"* (re-letters the page in about a second, no redrawing)
- *"Make panel 3 a close-up from a low angle"* (camera direction is editable)
- *"What is panel 4 about?"* — it can just answer questions, too.

**Attachments** — click the paperclip (or drag files onto the chat, or paste an image) to
attach **PDF, DOCX, PNG, JPG or SVG** files, up to 5 per message and 15 MB each:

| File | What the assistant does with it |
|---|---|
| PDF / DOCX | Reads the text as source material: *"rewrite the story using this brief"* |
| PNG / JPG | Can be used as a character's look: *"use this photo as the new sidekick"* |
| SVG | Reads its text labels **and** renders it, so it works as a character reference too |

Older `.doc` files aren't supported (save as `.docx`). Text inside attachments is treated
as reference material only — instructions hidden in a document are ignored.

**Only what changed is redrawn.** Text-only edits (dialogue, captions, title, reordering)
recompose the page in about a second with no GPU work. A scene change redraws that one panel
with the same seed and the character's existing portrait, so it still looks like the same
comic. Chat edits queue behind any drawing already in progress.

**Undo / Redo** — every edit is saved as a new version (up to `MAX_VERSIONS`, default 30),
so you can step back and forth with the buttons at the top of the chat. Comics and their chat
history survive a server restart and are restored when you reload the page.

The assistant uses the model selected in the main form, and works best with an instruction-tuned
model of 7B or larger (it must return valid JSON). If the model can see images (an Ollama
model with the *vision* capability), attached images are shown to it as well; otherwise it relies
on what you tell it about them. Everything the assistant did is reported back by the app from what
was actually applied — including anything it had to skip, and why.

Related settings in `.env`: `MAX_ATTACHMENTS`, `MAX_ATTACHMENT_MB`, `MAX_VERSIONS`.

---

## Comics craft knowledge

`backend/knowledge/comics_expert.md` (your comics-expert guide) is wired into the prompts by
`backend/expert.py`. It is written as a conversational system prompt, so it is not pasted blindly
into an app that needs strict JSON back. `COMICS_EXPERT` in `.env` chooses how it is used:

| Mode | What the model gets | Cost |
|---|---|---|
| `condensed` (default) | The rules that change what a script-writer does: story arc for the panel count (setup / development / twist / punchline, Establisher-Initial-Peak-Release), one drawable moment per panel, varied shots and angles, 180-degree rule, word/image economy, lettering awareness, anchor features | ~900 tokens |
| `full` | The entire guide, with its conversational output formats explicitly overridden by the JSON format | ~6.5k tokens, needs `OLLAMA_NUM_CTX >= 16384` |
| `off` | Nothing (original behaviour) | none |

The chat assistant always gets a short edit-time version. The guide also drives the new per-panel
`shot`, `angle` and `placement` fields, which feed both the image prompt and the lettering.
Edit the `.md` and restart to change the guide; no code changes needed.

---

## Switching image providers

Everything is env-driven (`config.py` reads `.env`):

| `IMAGE_PROVIDER` | Engine | Character refs? | VRAM |
|---|---|---|---|
| **`qwen`** (default) | Qwen-Image-2.1 | ✅ Strongest (native conditioning) | ~32 GB (offloaded) |
| **`local`** | SDXL + IP-Adapter | ✅ (blending) | ~12 GB |
| **`hf_inference`** | HF Inference API | ❌ | 0 (cloud) |

To switch, just change the line in `.env`:

```env
IMAGE_PROVIDER=qwen    # or "local" or "hf_inference"
```

Related tunables for the Qwen path:

```env
QWEN_MODEL_ID=Qwen/Qwen-Image-2.1
QWEN_STEPS=20            # 12–16 on smaller GPUs
QWEN_PANEL_SIZE=1024     # 768 on <24 GB VRAM
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torchvision'`** — install it:
  `pip install torchvision` (or the paired version with your torch).
- **`ImportError: Qwen3VLVideoProcessor requires the Torchvision library`** —
  same fix: install torchvision. This is the #1 error on a fresh env.
- **`torch` got downgraded after I installed torchvision** — expected, see the
  [torch downgrade note](#-torch-downgrade-note-windows--qwen-path) above.
- **First Qwen run is slow and I'm not sure it's working** — the model is ~33 GB.
  Download resumes safely across failures. To check progress:
  ```bash
  du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1
  ```
  When it hits ~33 GiB you're done.
- **`HF_HUB_OFFLINE=1`** — set this env var after the initial download to
  guarantee fully-offline operation. No network calls at run time.
- **numpy 2.x vs 1.x** — a few older ML packages pin `numpy<2.0`. If you hit
  an `incompatible` warning, it's harmless; the app itself doesn't depend on
  either.
- **CUDA out of memory** — lower `QWEN_PANEL_SIZE` to 768 and `QWEN_STEPS` to 12,
  or switch to `IMAGE_PROVIDER=local`.

---

## Project layout

```
.
├── backend/
│   ├── main.py          FastAPI app + job orchestration
│   ├── pipeline.py      The ONE drawing loop: cast references, then panels (all providers)
│   ├── render.py        Draw one portrait/panel; quality presets; reference prep; model preload
│   ├── prompts.py       Image-prompt builder (style first, camera, lettering headroom)
│   ├── styles.py        Art-style registry (prompt, negatives, lettering theme)
│   ├── expert.py        Comics-craft knowledge for the LLM + script QA checks
│   ├── knowledge/       comics_expert.md
│   ├── llm.py           Ollama / HF script generation, sanitising, identity helpers
│   ├── editor.py        Chat assistant: actions, applying them, redrawing only what changed
│   ├── imagegen.py      SDXL + IP-Adapter engine (IMAGE_PROVIDER=local) and hosted engine
│   ├── qwen_gen.py      Qwen-Image-2.1 engine (IMAGE_PROVIDER=qwen)
│   ├── characters.py    Character reference storage
│   ├── layout.py        Lettering engine + page layout (Pillow)
│   ├── config.py        All env-driven settings
│   ├── schemas.py       Pydantic models
│   └── assets/fonts/    Bangers (titles) and Comic Neue (lettering), both SIL OFL
├── frontend/            Vanilla HTML/CSS/JS UI, no build step
├── tools/
│   └── prefetch_models.py   One-time model pre-download + pipeline warm-up
 ├── requirements.txt     Python dependencies
 ├── requirements-dev.txt Dev deps (pytest) — only needed to run the local test suite
 ├── .env.example         Env-var template (copy to .env)
 ├── LICENSE              MIT
 └── README.md            You are here

 Not in the repo (git-ignored):
 ├── storage/             Uploaded character refs + generated comics (created at runtime)
 └── tests/               Local pytest suite (no GPU needed)
```
