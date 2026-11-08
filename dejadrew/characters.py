"""The storyboard's recurring characters, and how each one looks.

`build` asks who an illustrator must draw the same way in every picture,
then asks for one visual paragraph per character — typical clothing, not
a single scene. Both answers come back from the LLM cache when the
storyboard has not changed. Nothing in this module is written to disk.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Tier
from .llm import ask
from .schemas import (
    MONOCHROME_NOTE,
    ROLE,
    CharacterAppearance,
    CharacterList,
    CharacterRef,
    character_appearance_model,
)

TEXT_CHARS = 60_000


@dataclass
class Character:
    """One locked figure. `portrait` is set once the sheet has a path."""

    id: str
    name: str
    role: str
    appearance: str
    portrait: Path | None = None


def build(
    title: str | None,
    scenes: Sequence[str],
    *,
    tier: Tier,
    monochrome: bool = False,
) -> list[Character]:
    """The cast, each with an appearance paragraph. Cached when unchanged.

    When `monochrome` is true, each appearance is written for a
    monochromatic image and names no colours.
    """
    listed = list_characters(title, scenes, tier=tier)
    text = story_text(scenes)
    cast: list[Character] = []
    for ref in listed.characters:
        described = describe_appearance(title, text, ref, tier=tier, monochrome=monochrome)
        cast.append(
            Character(
                id=ref.id,
                name=ref.name,
                role=ref.role,
                appearance=described.appearance.strip(),
            )
        )
    return cast


_LIST_SYSTEM = (
    ROLE
    + "The same people must look like themselves from picture to picture. "
    "Your task is to list the characters that illustrator needs to lock in.\n\n"
    "Keep someone only if they appear in more than one scene, so the "
    "illustrator must draw them the same way later. Do not list anyone who "
    "appears in a single scene, even if that scene is about them. Drop "
    "unnamed crowds, one-line extras, and anyone who is only mentioned. An "
    "animal, spirit, or object counts if it has a face or a stable visual "
    "identity and recurs across scenes.\n\n"
    "Give each an English slug id the later steps will copy exactly.\n\n"
    "First, reason carefully about the storyboard. Then list the characters."
)

_APPEARANCE_SYSTEM = (
    ROLE
    + "An image model will draw this character many times, in different "
    "scenes, and must keep them looking like the same person. Your task is "
    "to write their generic appearance — the default look, not a moment "
    "from the plot.\n\n"
    "Write a short description that is easy to draw. Cover the big "
    "details a viewer notices at a glance, and leave out the rest. The "
    "goal is to make the drawing easier, not to specify every feature.\n\n"
    "Keep what stays the same from picture to picture and reads from "
    "across the room: a broad age (a child, a young woman, an old man), "
    "the build when it shows (tall and thin, small and round), hair or "
    "coat colour and its simple shape, the few usual clothes and their "
    "colours, and a mark only when it identifies them (a red cloak, a "
    "long white beard, a wooden leg, the colour of a horse). Use what "
    "the text says. When it is silent, infer only those large facts "
    "from the role, rank, era, and country.\n\n"
    "Omit small details, and anything that changes with time or from "
    "scene to scene: exact age, measurements, the shape of the eyes, "
    "brows, nose, mouth, or ears, skin texture, how the hair is parted, "
    "fabric, stitching, cut, condition, footwear, moles, rings, and "
    "small tools. Also omit dirt, tears, wounds, an expression, and "
    "whatever they happen to be holding. A plain face needs no "
    "inventory. Do not describe a pose, a setting, an action, or an "
    "artistic style.\n\n"
    "A few plain sentences are enough. Name a colour or a shape only "
    "when a second drawing would otherwise get that big fact wrong.\n\n"
    "First, decide which details are worth drawing. Then write the paragraph."
)


def _appearance_system(monochrome: bool) -> str:
    """Appearance instructions. The monochrome line is added only when asked."""
    if not monochrome:
        return _APPEARANCE_SYSTEM
    return f"{_APPEARANCE_SYSTEM}\n\n{MONOCHROME_NOTE}"


def story_text(scenes: Sequence[str]) -> str:
    """The whole storyboard as one string."""
    return "\n\n".join(scene.strip() for scene in scenes if scene and scene.strip())


def story_line(title: str | None) -> str:
    """`Story: "<title>"` and a blank line, or nothing when there is no title."""
    if not title or not title.strip():
        return ""
    return f"Story: {json.dumps(title.strip(), ensure_ascii=False)}\n\n"


def _scene_lines(scenes: Sequence[str]) -> str:
    """One scene per line, `Scene n: text`, with line breaks inside a scene removed."""
    lines: list[str] = []
    for n, scene in enumerate(scenes, start=1):
        flat = " ".join(scene.split())
        lines.append(f"Scene {n}: {flat}")
    return "\n".join(lines)


def cut(text: str) -> str:
    """`text`, cut to `TEXT_CHARS` with a note of how much was left out."""
    if len(text) <= TEXT_CHARS:
        return text
    extra = len(text) - TEXT_CHARS
    return f"{text[:TEXT_CHARS]}\n\n[... the text continues for {extra} more characters]"


def slug(value: str, fallback: str = "character") -> str:
    ascii_ = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", ascii_).strip("_")
    if not text:
        text = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_").lower()
    return text[:60] or fallback


def _unique_id(raw: str, taken: set[str], fallback: str) -> str:
    base = slug(raw, fallback)
    candidate = base
    n = 2
    while candidate in taken:
        candidate = f"{base}_{n}"
        n += 1
    taken.add(candidate)
    return candidate


def list_characters(title: str | None, scenes: Sequence[str], *, tier: Tier) -> CharacterList:
    """Recurring figures from the whole storyboard. Empty list if no model answers."""
    if isinstance(scenes, str):
        raise TypeError("scenes must be a sequence of strings, not a single string")
    result = ask(
        CharacterList,
        system=_LIST_SYSTEM,
        user=(
            f"{story_line(title)}"
            "The storyboard, one scene per line:\n"
            f"{cut(_scene_lines(scenes)) or '(no text)'}\n\n"
            "After your analysis, list the characters."
        ),
        tier=tier,
    )
    if result is None:
        return CharacterList(analysis="No model answered; no characters locked in.", characters=[])
    taken: set[str] = set()
    cleaned: list[CharacterRef] = []
    for row in result.characters:
        ident = _unique_id(row.id or row.name, taken, f"character_{len(cleaned) + 1}")
        name = row.name.strip() or ident
        cleaned.append(CharacterRef(id=ident, name=name, role=row.role.strip()))
    return CharacterList(analysis=result.analysis, characters=cleaned)


def describe_appearance(
    title: str | None,
    text: str,
    character: CharacterRef,
    *,
    tier: Tier,
    monochrome: bool = False,
) -> CharacterAppearance:
    """One visual paragraph for `character`. Generic line if no model answers.

    When `monochrome` is true, the paragraph is for a monochromatic image
    and must not name colours.
    """
    closing = (
        "After your analysis, write this character's generic appearance. "
        "A few plain sentences: the big, stable, eye-catching facts only, "
        "such as age in broad terms, build, hair or coat colour, the "
        "usual clothes and their colours, and a mark that identifies "
        "them. Leave out small features and anything that changes."
    )
    if monochrome:
        closing = f"{closing}\n\n{MONOCHROME_NOTE}"
    result = ask(
        character_appearance_model(monochrome),
        system=_appearance_system(monochrome),
        user=(
            f"{story_line(title)}"
            f"Character id: {character.id}\n"
            f"Name: {character.name}\n"
            f"Role: {character.role or '(unspecified)'}\n\n"
            f"The storyboard:\n{cut(text) or '(no text)'}\n\n"
            f"{closing}"
        ),
        tier=tier,
    )
    if result is not None and result.appearance.strip():
        return result
    who = character.name or character.id
    return CharacterAppearance(
        analysis="No model answered; appearance inferred from the name and role.",
        appearance=f"An adult figure known as {who}" + (f", {character.role}" if character.role else "") + ".",
    )
