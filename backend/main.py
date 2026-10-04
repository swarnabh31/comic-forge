import asyncio
import io
import logging
import time
import uuid

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

import attachments
import characters
import comic_store
import config
import editor
import layout
import llm
import pipeline
import render
import styles
from schemas import Character, GenerateRequest, JobStatus

log = logging.getLogger("comicforge")

app = FastAPI(title="Comic Forge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, JobStatus] = {}
ACTIVE_EDITS: set[str] = set()        # comic ids with a chat request currently being planned/applied
_TASKS: set = set()                   # keeps background tasks referenced until they finish


@app.get("/api/config")
def get_config():
    return {
        "llm_provider": config.LLM_PROVIDER, "image_provider": config.IMAGE_PROVIDER, "max_panels": config.MAX_PANELS,
        "default_text_size": config.DEFAULT_TEXT_SIZE, "default_quality": config.DEFAULT_QUALITY,
        "default_style": styles.DEFAULT_STYLE_ID,
    }


@app.get("/api/styles")
def get_styles():
    """The art styles the picker offers (id, label, group, one-line description)."""
    return {"default": styles.DEFAULT_STYLE_ID, "styles": styles.catalogue()}


@app.get("/api/models")
def get_models():
    if config.LLM_PROVIDER != "ollama":
        return {"models": [config.HF_LLM_MODEL]}
    try:
        return {"models": llm.list_ollama_models()}
    except Exception as exc:  # Ollama not running, wrong host, etc.
        raise HTTPException(502, f"Couldn't reach Ollama at {config.OLLAMA_HOST}: {exc}")


def _char_view(c: Character) -> dict:
    """Character + browser-usable image URLs (the stored paths are absolute filesystem paths)."""
    d = c.model_dump()
    d["image_urls"] = [f"/characters/{Path(p).parent.name}/{Path(p).name}" for p in c.reference_images]
    return d


@app.get("/api/characters")
def get_characters():
    return [_char_view(c) for c in characters.list_characters()]


@app.post("/api/characters")
async def add_character(name: str = Form(...), description: str = Form(""), files: list[UploadFile] = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Give the character a name.")
    if len(characters.list_characters()) >= config.MAX_CHARACTERS:
        raise HTTPException(400, f"You can keep at most {config.MAX_CHARACTERS} characters. Remove one first.")
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    contents, names = [], []
    for f in files:
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(400, f"{f.filename} is over {config.MAX_UPLOAD_MB} MB.")
        try:                                   # a renamed .txt must not become a "reference image"
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except Exception:
            raise HTTPException(400, f"{f.filename} isn't a readable image.")
        contents.append(data)
        names.append(f.filename or "ref.png")
    created = characters.create_character(name, description.strip(), contents, names)
    # If the user didn't describe the character, look at their own reference image and derive a literal,
    # name-free appearance description via the vision LLM. This becomes the canonical identity the comic is
    # drawn from: the image model is shown real features instead of a raw name like "Mickey".
    if not created.description and created.reference_images:
        vision_model = llm.pick_vision_model()
        if vision_model:
            derived = (await run_in_threadpool(llm.describe_reference, created.reference_images[0], vision_model)).strip()
            if derived:
                created = characters.set_description(created.id, derived) or created
    return _char_view(created)


@app.patch("/api/characters/{character_id}")
def update_character(character_id: str, description: str = Form("")):
    """Update a character's name/description. The description is the canonical
    identity used when drawing the character (the raw name is stripped from the
    image prompt, so the description IS the visual cue the model sees)."""
    updated = characters.set_description(character_id, description)
    if updated is None:
        raise HTTPException(404, "Unknown character id")
    return updated


@app.get("/api/characters/{character_id}/describe")
async def describe_character(character_id: str, model: str = ""):
    """Re-derive a character's appearance description from their uploaded image.
    Returns just the description text; the frontend uses it to pre-fill the
    description field, which it then submits with the normal add-character form."""
    c = characters.get_character(character_id)
    if not c:
        raise HTTPException(404, "Unknown character id")
    if not c.reference_images:
        raise HTTPException(400, "No reference image to describe.")
    vision_model = model or llm.pick_vision_model()
    if not vision_model:
        raise HTTPException(502, "No vision-capable Ollama model found on this machine.")
    text = (await run_in_threadpool(llm.describe_reference, c.reference_images[0], vision_model)).strip()
    if not text:
        raise HTTPException(502, "The vision model returned no description.")
    return {"name": c.name, "description": text}


@app.delete("/api/characters/{character_id}")
def remove_character(character_id: str):
    characters.delete_character(character_id)
    return {"ok": True}


def _run_job(job_id: str, req: GenerateRequest):
    job = JOBS[job_id]
    try:
        style = styles.resolve(req.art_style)
        art_style = styles.canonical(req.art_style)
        uploaded = [characters.get_character(cid) for cid in req.character_ids]
        uploaded_by_name = {c.name: c for c in uploaded if c}
        cast_info = [{"name": c.name, "description": c.description or "", "is_person": True} for c in uploaded if c]

        # Hide the 20-30 s image-model load behind the LLM's script writing.
        render.preload_async()

        job.state, job.message, job.progress = "writing_script", "Writing the comic script...", 0.03
        script, notes = llm.generate_script_with_notes(req.prompt, req.model, req.panel_count, cast_info, req.text_size)
        job.script = script
        job.warnings.extend(notes)
        job.progress = 0.09
        if config.UNLOAD_LLM_BEFORE_DRAWING:
            llm.unload(req.model)                 # frees VRAM for the image model

        identities = pipeline.build_identities(script, uploaded_by_name, req.model)
        seed = render.new_seed()
        cdir = config.COMICS_DIR / job_id
        cdir.mkdir(parents=True, exist_ok=True)

        def progress(msg: str, frac: float) -> None:
            job.state, job.message, job.progress = "drawing", msg, frac

        def on_panel(i: int, img) -> None:       # show each panel in the UI the moment it exists
            img.save(cdir / f"panel_{i + 1}.png")
            job.panel_urls.append(f"/comics/{job_id}/panel_{i + 1}.png")

        drawn = pipeline.draw_comic(script, art_style, req.quality, req.text_size, uploaded_by_name, identities,
                                    seed, progress, on_panel)
        job.warnings.extend(drawn.warnings)

        job.state, job.message, job.progress = "compositing", "Lettering and laying out the page...", 0.93
        page, layout_notes = layout.compose_page_with_notes(script, drawn.panel_images, style, req.text_size)
        job.warnings.extend(layout_notes)
        page.save(cdir / "page.png")
        job.page_url = f"/comics/{job_id}/page.png"

        # Persist everything the chat assistant needs to edit this comic later. A failure here must not lose the
        # finished comic: the chat just won't be offered for it.
        try:
            _save_comic_state(job_id, req, art_style, script, drawn, seed, uploaded_by_name)
        except Exception:
            log.exception("Couldn't save chat state for comic %s", job_id)

        job.state, job.message, job.progress = "done", "Your comic is ready!", 1.0
    except Exception as exc:
        log.exception("Comic job %s failed", job_id)
        job.state, job.error, job.message = "error", str(exc) or type(exc).__name__, "Something went wrong."


def _spawn(fn, *args) -> None:
    task = asyncio.get_running_loop().create_task(run_in_threadpool(fn, *args))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def _run_job_locked(job_id: str, req: GenerateRequest):
    """All image work shares one GPU pipeline, so jobs (new comics and chat edits) take turns."""
    job = JOBS[job_id]
    if not render.GPU_LOCK.acquire(blocking=False):
        job.message = "Waiting for another drawing job to finish..."
        render.GPU_LOCK.acquire()
    try:
        _run_job(job_id, req)
    finally:
        render.GPU_LOCK.release()


def _save_comic_state(job_id, req, art_style, script, drawn, seed, uploaded_by_name):
    cdir = config.COMICS_DIR / job_id
    refs = {}
    for name, imgs in drawn.refs.items():
        if not imgs:
            continue
        rel = f"refs/{comic_store.safe_name(name)}_{comic_store.new_uid()}.png"
        (cdir / rel).parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(cdir / rel)
        entry = {"file": rel, "source": "uploaded" if name in uploaded_by_name else "generated"}
        if name in drawn.originals:           # keep the user's own picture so a later style change can restyle it again
            orig = f"refs/{comic_store.safe_name(name)}_orig_{comic_store.new_uid()}.png"
            drawn.originals[name].save(cdir / orig)
            entry["original"] = orig
        refs[name] = entry
    plates = {}
    for lid, img in drawn.plates.items():
        rel = f"refs/plate_{comic_store.safe_name(lid)}_{comic_store.new_uid()}.png"
        (cdir / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(cdir / rel)
        plates[lid] = rel
    snapshot = {
        "art_style": art_style,
        "plates": plates,
        "text_size": req.text_size,
        "quality": req.quality,
        "script": script.model_dump(),
        "panels": [
            {"uid": comic_store.new_uid(), "file": f"panel_{i + 1}.png", "seed": drawn.seeds[i] if i < len(drawn.seeds) else seed + i + 1}
            for i in range(len(drawn.panel_images))
        ],
        "refs": refs,
        "page": "page.png",
    }
    comic_store.create(job_id, model=req.model, seed=seed, snapshot=snapshot)


JOB_TTL_SECONDS = 6 * 3600


def _prune_jobs() -> None:
    """JOBS used to grow forever (each holds a script and URLs). Drop finished jobs after a few hours."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for jid in [j for j, job in JOBS.items() if job.state in ("done", "error") and JOB_BORN.get(j, cutoff + 1) < cutoff]:
        JOBS.pop(jid, None)
        JOB_BORN.pop(jid, None)


JOB_BORN: dict[str, float] = {}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    req.prompt = req.prompt.strip()
    if not req.prompt:
        raise HTTPException(400, "Type an idea for the comic first.")
    if config.LLM_PROVIDER == "ollama" and not req.model:
        raise HTTPException(400, "No language model selected. Is Ollama running?")
    req.panel_count = max(1, min(req.panel_count, config.MAX_PANELS))
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = JobStatus(id=job_id, state="queued")
    JOB_BORN[job_id] = time.time()
    _spawn(_run_job_locked, job_id, req)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JobStatus:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    return job


# ------------------------------------------------------------------ chat assistant

def _comic_or_404(comic_id: str) -> dict:
    if not comic_store.exists(comic_id):
        raise HTTPException(404, "Unknown comic id")
    return comic_store.load(comic_id)


@app.get("/api/comics/{comic_id}")
def get_comic(comic_id: str):
    state = _comic_or_404(comic_id)
    return {**comic_store.view(state), "busy": comic_id in ACTIVE_EDITS}


@app.post("/api/comics/{comic_id}/chat")
async def chat_with_comic(
    comic_id: str,
    message: str = Form(""),
    model: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    """One chat turn. The LLM plans the edit here (a few seconds); the actual drawing runs as a
    background job the browser polls via /api/jobs/{job_id}."""
    message = message.strip()
    files = [f for f in files if f.filename]
    if not message and not files:
        raise HTTPException(400, "Type a message or attach a file.")
    if len(files) > config.MAX_ATTACHMENTS:
        raise HTTPException(400, f"Attach at most {config.MAX_ATTACHMENTS} files per message.")
    _comic_or_404(comic_id)
    if config.LLM_PROVIDER == "ollama" and not model:
        raise HTTPException(400, "Pick a model in the main form first — the assistant uses the same one.")
    if comic_id in ACTIVE_EDITS:
        raise HTTPException(409, "I'm still working on your last request — one moment.")

    ACTIVE_EDITS.add(comic_id)
    handed_off = False
    try:
        limit = config.MAX_ATTACHMENT_MB * 1024 * 1024
        parsed = []
        for f in files:
            data = await f.read(limit + 1)
            try:
                parsed.append(await run_in_threadpool(attachments.process, f.filename, data, comic_id))
            except attachments.AttachmentError as exc:
                raise HTTPException(400, str(exc))
        new_atts = comic_store.add_attachments(comic_id, parsed) if parsed else []

        state = comic_store.load(comic_id)
        try:
            reply, actions, warnings = await run_in_threadpool(editor.plan, state, message, new_atts, model)
        except Exception as exc:
            log.exception("Chat planning failed")
            reply = f"I couldn't get an answer from the language model ({type(exc).__name__}: {str(exc)[:160]}). Is it running?"
            actions, warnings = [], []

        att_refs = [{"id": a["id"], "name": a["name"], "kind": a["kind"]} for a in new_atts]
        comic_store.append_chat(comic_id, [
            {"role": "user", "kind": "message", "content": message, "attachments": att_refs},
            {"role": "assistant", "kind": "message", "content": reply, "actions": [a.model_dump(exclude_none=True) for a in actions]},
        ])

        job_id = None
        if actions:
            job_id = uuid.uuid4().hex[:12]
            JOBS[job_id] = JobStatus(id=job_id, state="queued", message="Getting started...")
            JOB_BORN[job_id] = time.time()
            label = (message or "Edit")[:60]

            def _edit(job_id=job_id, label=label):
                try:
                    editor.run_edit_job(JOBS[job_id], comic_id, actions, label)
                finally:
                    ACTIVE_EDITS.discard(comic_id)

            _spawn(_edit)
            handed_off = True
        return {"reply": reply, "warnings": warnings, "job_id": job_id, "attachments": att_refs}
    finally:
        if not handed_off:
            ACTIVE_EDITS.discard(comic_id)


def _step(comic_id: str, delta: int):
    _comic_or_404(comic_id)
    if comic_id in ACTIVE_EDITS:
        raise HTTPException(409, "Wait for the current edit to finish first.")
    state, note = comic_store.step(comic_id, delta)
    return {**comic_store.view(state), "note": note}


@app.post("/api/comics/{comic_id}/undo")
def undo_comic(comic_id: str):
    return _step(comic_id, -1)


@app.post("/api/comics/{comic_id}/redo")
def redo_comic(comic_id: str):
    return _step(comic_id, +1)


@app.post("/api/comics/{comic_id}/text_size")
async def set_comic_text_size(comic_id: str, text_size: str = Form(...)):
    """Re-letter the comic at a new text size without redrawing anything (about a second)."""
    if text_size not in ("small", "medium", "large"):
        raise HTTPException(400, "text_size must be small, medium or large.")
    _comic_or_404(comic_id)
    if comic_id in ACTIVE_EDITS:
        raise HTTPException(409, "Wait for the current edit to finish first.")
    ACTIVE_EDITS.add(comic_id)
    try:
        state = await run_in_threadpool(editor.set_text_size, comic_id, text_size)
    finally:
        ACTIVE_EDITS.discard(comic_id)
    return comic_store.view(state)


app.mount("/comics", StaticFiles(directory=str(config.COMICS_DIR)), name="comics")
app.mount("/characters", StaticFiles(directory=str(config.CHARACTERS_DIR)), name="characters")
app.mount("/", StaticFiles(directory=str(config.BASE_DIR / "frontend"), html=True), name="frontend")
