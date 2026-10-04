# What changed and why

## Requested features
1. **Text size (Small / Medium / Large)**: new layout engine (`layout.py`), word budgets for the writer, reserved headroom for the artist, re-letter-in-one-second after the fact, chat command.
2. **Comics expert guide**: `backend/knowledge/comics_expert.md` + `expert.py` (condensed / full / off).
3. **Art styles**: 5 loose phrases became 22 full style specs in `styles.py`, with consistent handling (see below).
4. **Evaluation**: findings and fixes below.

## Why the old styles weren't reliable
| Problem | Fix |
|---|---|
| Style was one phrase pasted at the END of the prompt; SDXL cuts after ~77 tokens, so long scenes lost it | Style block now goes FIRST (`prompts.py`) |
| One negative prompt for all styles, including "3D render" and "photo" even for a 3D style, and nothing against colour for B&W styles | Per-style negatives (`styles.negative_prompt`) |
| Uploaded photos/anime pictures were used as references as-is, so their own style leaked into the panels | Uploads are redrawn once in the chosen style (`render.restyle_reference`), original kept so a later style change restyles again |
| Switching style in chat only redrew generated portraits; uploads kept the old look | Uploads are restyled on style change |
| Lettering looked the same in every style | Each style has a lettering theme (paper, caption colour, caps) |
| Chat said "switch to anime" and the model had to invent a prompt | Style ids are given to the assistant; loose words ("anime", "watercolour") resolve to a style |

## Bugs and flawed logic fixed
- **Raw cast names reached the image model on the SDXL path** (the identity rewrite existed only in Qwen/HF loops). Three hand-copied drawing loops are now one (`pipeline.py`).
- `IMAGE_PROVIDER` defaulted to `local` in code while the README said `qwen`.
- Ollama was called without `num_ctx`; its small default window silently truncates the START of a long prompt (the instructions).
- JSON extraction used a greedy `{.*}` regex: broke on `<think>` blocks (qwen3) and on stray braces. Now a real decoder scan.
- A single malformed field from the LLM (caption as a list, `"null"`, bare-string dialogue, unknown speaker) failed the whole job after the wait. Now sanitised, retried once, and anything dropped is reported to the user.
- Constrained decoding now uses a schema that REQUIRES every field.
- Speaker normalisation dropped all but the first line regardless of speaker; now keeps the first speaker's lines.
- Page grid left empty cells for 5/7 panels; now row templates with centred short rows.
- Bubble drawing: default Pillow font, text wider than the bubble possible, no collision handling, jagged edges. Replaced.
- Phone photos arrived sideways (EXIF) and were huge; now rotated and shrunk before conditioning.
- Character names were inserted with `innerHTML` (HTML injection); upload accepted any file; character image URLs depended on a Windows path regex. Fixed.
- `JOBS` grew forever; finished jobs are pruned after 6 hours.
- `prefetch_models.py` and README referenced a `.env.example` that did not exist; added.

## Speed
- Image model loads in a background thread while the LLM writes the script (hides ~20-30 s).
- LLM is unloaded from VRAM before drawing (`UNLOAD_LLM_BEFORE_DRAWING`).
- Quality presets: Draft is roughly 2.5x faster.
- Character portraits drawn at 75% size; reference photos capped at 768 px.
- Panels appear in the UI as they finish.
- Text-size changes and text-only edits cost about a second and no GPU.

## Not changed (worth knowing)
- Panels are still square. Varying panel size and shape (wide/tall) is the biggest remaining comics-craft upgrade; the shot/angle fields are already in place to drive it.
- With several characters in one panel, identity blending remains a model limitation; one speaker per panel is still enforced.
- All numbers above for speed are estimates from the settings; measure on your hardware.


---

# Round 2: environment, positions and perspective (fixing my own mistakes)

Symptom: each panel showed a different room, props and characters moved around, and perspective was off.

## Root cause (pre-existing, not fixed in round 1)
Every panel was an independent image generated from its own text. Nothing defined the room, and every panel used a
different seed, so each started from a different layout. The comics guide (section 10) says to lock a style block, a
character block AND a location block. Only the first two existed.

## Mistakes I made in round 1, now fixed
| Mistake | Effect | Fix |
|---|---|---|
| Added `dutch tilt` / `bird's-eye` angles (and the guide pushed variety) | Image models have no 3D understanding; tilted/extreme views break perspective | Writer told to use eye level almost always; the prompt builder clamps tilted/bird's-eye/worm's-eye to eye level (`ALLOW_EXTREME_ANGLES=1` to allow) |
| Writer chose `placement` left/center/right and the prompt enforced it | The character jumped around the frame panel to panel | Placement is always center unless you change it in chat |
| "Top 40% empty, head well below the top edge" | Shrank the subject, invented a blank wall, broke scale and horizon | 12 / 18 / 25% by text size, worded as "plain wall/ceiling/sky", dropped entirely for close-ups; lettering is placed by the layout engine anyway |
| Portraits were head-focused reference sheets | No body proportions to condition on | Full body, head to feet, neutral pose |

## New: location bible and plates
- The script now defines 1-3 **locations** (layout, fixed props and where they are, lighting) and every panel names one. Panel text describes only what changes (pose, expression, props handled).
- The setting text is inserted **verbatim** into every panel of that location.
- All panels in a location share **one seed**, so they start from the same layout.
- Qwen engine: one empty **plate** picture per location is drawn and passed to every panel as a reference ("keep the same layout, furniture, props, lighting"). `LOCATION_PLATES=0` turns this off.
- SDXL/IP-Adapter engine: no plate (IP-Adapter would blend the room into the character's face); it gets the text block and the shared seed only.
- Chat: `update_location`, `add_location`, and `location` on panels. Adding a panel keeps it in the neighbouring room and seed. Changing the art style redraws the plates too.
- Honest limits: I could not run Qwen-Image here. The text block, shared seed and clamped angles are deterministic and tested; whether Qwen follows the plate as strongly as hoped needs one real comic to confirm.
