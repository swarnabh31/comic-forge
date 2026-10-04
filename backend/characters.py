"""Stores each character's uploaded reference images plus a short text
description, so panel generation can look them up by name and condition
the image model on their actual likeness."""
import json
import uuid
from pathlib import Path

import config
from schemas import Character

REGISTRY_PATH = config.CHARACTERS_DIR / "registry.json"


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def list_characters() -> list[Character]:
    return [Character.model_validate(c) for c in _load().values()]


def get_character(character_id: str) -> Character | None:
    data = _load().get(character_id)
    return Character.model_validate(data) if data else None


def create_character(name: str, description: str, image_bytes_list: list[bytes], filenames: list[str]) -> Character:
    registry = _load()
    char_id = uuid.uuid4().hex[:10]
    char_dir: Path = config.CHARACTERS_DIR / char_id
    char_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for content, filename in zip(image_bytes_list, filenames):
        suffix = Path(filename).suffix or ".png"
        dest = char_dir / f"ref_{len(saved_paths)}{suffix}"
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    character = Character(id=char_id, name=name, description=description, reference_images=saved_paths)
    registry[char_id] = character.model_dump()
    _save(registry)
    return character


def delete_character(character_id: str) -> None:
    registry = _load()
    registry.pop(character_id, None)
    _save(registry)


def set_description(character_id: str, description: str) -> Character | None:
    """Update a character's appearance description (the canonical identity used
    when drawing). Returns the updated character, or None if it doesn't exist."""
    registry = _load()
    if character_id not in registry:
        return None
    registry[character_id]["description"] = description
    _save(registry)
    return Character.model_validate(registry[character_id])
