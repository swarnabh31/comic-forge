from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

TextSize = Literal["small", "medium", "large"]
Quality = Literal["draft", "standard", "high"]
Placement = Literal["left", "center", "right"]


class DialogueLine(BaseModel):
    speaker: str
    text: str
    type: Literal["speech", "thought"] = "speech"

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        return "thought" if isinstance(v, str) and "thought" in v.lower() else "speech"


class Panel(BaseModel):
    index: int
    scene_description: str
    characters: list[str] = Field(default_factory=list)
    caption: Optional[str] = None
    dialogue: list[DialogueLine] = Field(default_factory=list)
    # Camera direction (comics-craft: vary the shot, state the angle, say where the subject sits so
    # the lettering can be planned around them). All optional so older comics still load.
    shot: str = ""            # e.g. "wide establishing", "medium", "close-up", "insert"
    angle: str = ""           # e.g. "eye level", "low angle", "high angle", "dutch tilt"
    placement: Placement = "center"   # where the speaker stands in the frame (center unless the user moves them)
    location: str = ""                # id of a Location in the script; "" = none

    @field_validator("placement", mode="before")
    @classmethod
    def _placement(cls, v):
        s = str(v or "").lower()
        return "left" if "left" in s else "right" if "right" in s else "center"

    @field_validator("shot", "angle", mode="before")
    @classmethod
    def _short_text(cls, v):
        return str(v or "").strip()[:60]


class CastMember(BaseModel):
    """A recurring character or named entity (a person, pet, or even a talking
    appliance) that needs the SAME visual identity in every panel it appears in."""
    name: str
    appearance: str
    is_person: bool = True


class Location(BaseModel):
    """A place the story happens in, described ONCE so every panel set there shows the same room."""
    id: str
    description: str          # layout, fixed props and where they are, lighting, time of day, colour scheme


class ComicScript(BaseModel):
    title: str
    cast: list[CastMember] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    panels: list[Panel]


class GenerateRequest(BaseModel):
    prompt: str
    model: str
    panel_count: int = 6
    # A style id from /api/styles (e.g. "manga_bw"), or any free-text style phrase.
    art_style: str = "classic_superhero"
    text_size: TextSize = "medium"
    quality: Quality = "standard"
    character_ids: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    id: str
    state: Literal["queued", "writing_script", "drawing", "compositing", "done", "error"]
    progress: float = 0.0
    message: str = ""
    script: Optional[ComicScript] = None
    page_url: Optional[str] = None
    panel_urls: list[str] = Field(default_factory=list)   # filled live, as each panel finishes
    warnings: list[str] = Field(default_factory=list)     # non-fatal notes about the result
    error: Optional[str] = None
    # Set on chat-edit jobs:
    summary: Optional[str] = None      # what was actually applied (written by the app, not the LLM)
    version: Optional[int] = None      # comic version created by the edit


class Character(BaseModel):
    id: str
    name: str
    description: str = ""
    reference_images: list[str] = Field(default_factory=list)
