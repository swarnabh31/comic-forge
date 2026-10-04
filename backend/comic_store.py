"""On-disk state for every generated comic, so the chat assistant can edit it
later (even after a server restart).

Layout:  storage/comics/<comic_id>/state.json  + the PNGs it points to.

A comic is a list of immutable *versions* (snapshots). Every chat edit appends
a new snapshot and moves `current` to it, which is what makes Undo / Redo cheap:
it only moves the pointer, nothing is redrawn. Image files are never
overwritten, they get fresh names, so old versions stay intact.

    state = {
      id, model, seed,                   # seed: the comic's base seed (portraits + style stability)
      versions: [snapshot, ...], current: int, next_version: int,
      chat: [ {role, content, kind, actions?, attachments?} ],
      attachments: { "A1": {name, kind, meta, text, image_file, ...} },
    }
    snapshot = {
      version, label, art_style (style id or custom phrase), text_size, quality,
      script:  ComicScript dict,
      panels:  [ {uid, file, seed} ]     # parallel to script["panels"]
      refs:    { name: {file, source} }  # source: uploaded | generated | attachment
      page:    "page.png"
    }
"""
import json
import os
import re
import threading
import uuid
from pathlib import Path

import config

COMIC_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_LOCK = threading.RLock()


def comic_dir(comic_id: str) -> Path:
    if not COMIC_ID_RE.match(comic_id):
        raise KeyError(comic_id)
    return config.COMICS_DIR / comic_id


def _path(comic_id: str) -> Path:
    return comic_dir(comic_id) / "state.json"


def exists(comic_id: str) -> bool:
    try:
        return _path(comic_id).exists()
    except KeyError:
        return False


def load(comic_id: str) -> dict:
    with _LOCK:
        return json.loads(_path(comic_id).read_text(encoding="utf-8"))


def _save(state: dict) -> None:
    path = _path(state["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)          # atomic: a crash never leaves half a state file


def current(state: dict) -> dict:
    return state["versions"][state["current"]]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:40] or "x"


def new_uid() -> str:
    return uuid.uuid4().hex[:8]


def create(comic_id: str, *, model: str, seed: int, snapshot: dict) -> dict:
    snapshot = {**snapshot, "version": 1, "label": "Original"}
    state = {
        "id": comic_id, "model": model, "seed": seed,
        "versions": [snapshot], "current": 0, "next_version": 2,
        "chat": [], "attachments": {},
    }
    with _LOCK:
        _save(state)
    return state


def push_version(comic_id: str, snapshot: dict, label: str) -> dict:
    """Append a new version after `current`. Anything that was 'redo-able' is dropped."""
    with _LOCK:
        state = load(comic_id)
        snapshot = {**snapshot, "version": state["next_version"], "label": label[:80]}
        state["next_version"] += 1
        state["versions"] = state["versions"][: state["current"] + 1] + [snapshot]
        if len(state["versions"]) > config.MAX_VERSIONS:
            state["versions"] = state["versions"][-config.MAX_VERSIONS:]
        state["current"] = len(state["versions"]) - 1
        _save(state)
        return state


def step(comic_id: str, delta: int) -> tuple[dict, str | None]:
    """Undo (-1) / redo (+1). Returns (state, note) — note is None if there was nowhere to go."""
    with _LOCK:
        state = load(comic_id)
        target = state["current"] + delta
        if not 0 <= target < len(state["versions"]):
            return state, None
        state["current"] = target
        v = state["versions"][target]["version"]
        note = f"Went {'back' if delta < 0 else 'forward'} to version {v}."
        state["chat"].append({"role": "assistant", "kind": "result", "content": note})
        _save(state)
        return state, note


def append_chat(comic_id: str, entries: list[dict]) -> None:
    with _LOCK:
        state = load(comic_id)
        state["chat"].extend(entries)
        _save(state)


def add_attachments(comic_id: str, metas: list[dict]) -> list[dict]:
    """Give each parsed upload a per-comic id (A1, A2, ...) and remember it."""
    with _LOCK:
        state = load(comic_id)
        out = []
        for meta in metas:
            att_id = f"A{len(state['attachments']) + 1}"
            state["attachments"][att_id] = {**meta, "id": att_id}
            out.append(state["attachments"][att_id])
        _save(state)
        return out


def view(state: dict) -> dict:
    """What the browser needs to show the current version."""
    snap = current(state)
    cid = state["id"]
    return {
        "id": cid,
        "version": snap["version"],
        "label": snap.get("label", ""),
        "title": snap["script"]["title"],
        "art_style": snap["art_style"],
        "text_size": snap.get("text_size", config.DEFAULT_TEXT_SIZE),
        "quality": snap.get("quality", config.DEFAULT_QUALITY),
        "page_url": f"/comics/{cid}/{snap['page']}",
        "panel_urls": [f"/comics/{cid}/{p['file']}" for p in snap["panels"]],
        "script": snap["script"],
        "can_undo": state["current"] > 0,
        "can_redo": state["current"] < len(state["versions"]) - 1,
        "chat": state["chat"][-200:],
    }
