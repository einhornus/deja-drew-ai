"""Who is in one scene, then the words the image model draws from.

Two calls per scene. The first names who from the locked cast is actually
visible, from a window of scenes around this one. The second writes the
scene prompt. That call sees, in order: the rules, every image prompt
already written for an earlier scene, the whole storyboard, and the scene
to illustrate. Each part is commented so the model knows what it is.
The prompt describes the scene only. Looks and style are attached later,
by `image_request`: the storyboard's style reference, then each chosen
character's portrait, then the drawing rules, then the artistic style.
The scene prompt is last, because it is the picture.

One earlier call, `choose_iconic_scene`, reads the whole storyboard and
names the moment drawn once as that style reference. Its prompt call uses
the same shape, with no scene number and no earlier prompts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from rapidfuzz import fuzz

from .characters import Character, cut, story_line, story_text
from .config import Tier
from .llm import ask
from .schemas import MONOCHROME_NOTE, ROLE, IconicScene, SceneCast, ScenePrompt, scene_prompt_model

CONTEXT_CHARS = 4000
MAX_REFERENCE_PORTRAITS = 6

RULES = (
    "Keep the picture simple and clean: few elements, a calm uncluttered "
    "background, one light. Whatever the scene leaves unsaid is yours to "
    "choose freely, and a clean picture matters more than a literal one — "
    "simplify or drop anything you cannot draw convincingly.\n"
    "No text of any kind anywhere in the image: no letters, words, numbers, "
    "titles, captions, speech bubbles, signatures, monograms, logos, or "
    "watermarks, and no pseudo-writing on signs, books, or banners."
)

_CAST_SYSTEM = (
    ROLE
    + "A character pool for this storyboard has already been locked in. "
    "Your task is to say which of those characters are physically present "
    "and would be visible in one picture of this scene.\n\n"
    "Copy ids from the pool. Do not invent ids. Leave the list empty if "
    "the picture is a place, an object, or a crowd of extras with no one "
    "from the pool. Mentioned-only or offstage characters stay out.\n\n"
    "First, reason carefully about the scene. Then list the ids."
)

_ICONIC_SYSTEM = (
    ROLE
    + "One picture will be the style reference for every illustration in "
    "this storyboard. Your task is to choose a single iconic scene that "
    "puts as many of the main characters as possible in that one picture "
    "together.\n\n"
    "Pick a moment a viewer would recognise: a key scene, a gathering, "
    "a confrontation, a feast, a departure. Everyone you name is physically "
    "present and visible, not merely mentioned. Name them as the pool "
    "names them. Prefer a clear setting and one action. Do not describe "
    "clothing, artistic style, or how to draw the picture: a later step "
    "writes the image prompt and casts the characters.\n\n"
    "First, reason carefully about the storyboard. Then describe the scene."
)

_PROMPT_SYSTEM = (
    ROLE
    + "An image model will draw this one scene. Your task is to write the "
    "prompt that model draws from.\n\n"
    "The user message is in parts. Each part starts with a # heading and "
    "# comments that say what the part is and how to use it. Read the "
    "comment before the part.\n\n"
    "The prompt is a detailed description of the scene itself: who is "
    "there, what each person is doing, where they are, the objects that "
    "matter, and how the moment is arranged. Nothing else. Do not describe "
    "how anyone looks, and do not give any style clue — no artistic style, "
    "medium, palette, line, texture, rendering, or 'in the style of'. "
    "Looks and style are attached later, and a style word here fights them.\n\n"
    "Name the characters as the rules name them. This picture is a new "
    "moment: do not repeat a picture already written for an earlier scene. "
    "Scenes that sit close together need more than a small change. When "
    "a nearby scene describes almost the same thing, depict it differently "
    "enough that the two pictures do not look like variants of one shot: "
    "another viewpoint, distance, arrangement, or emphasis. The closer the "
    "scenes, the more the pictures should differ.\n\n"
    "Image models refuse prompts that look violent, gory, sexual, or that "
    "put minors in danger, or that name real people or trademarked "
    "characters. If a prompt was refused, tell the same moment by "
    "suggestion and mood instead.\n\n"
    "First, reason carefully about the scene. Then write the prompt."
)


def _prompt_system(monochrome: bool) -> str:
    """Scene-prompt instructions. The monochrome line is added only when asked."""
    if not monochrome:
        return _PROMPT_SYSTEM
    return f"{_PROMPT_SYSTEM}\n\n{MONOCHROME_NOTE}"


# What the prompt-writer is told, before any text of the storyboard.
# `RULES` is the same constraint the image model sees later. The rest
# tells this call what to write, and what to leave out.
_SCENE_RULES = (
    "Write a detailed prompt for an image model. Describe only the scene: "
    "who is there, what each person is doing, where they are, the objects "
    "that matter, and how the moment is arranged in the frame. As long as "
    "the moment needs, and no longer.\n"
    "No style clues of any kind. Do not mention artistic style, medium, "
    "palette, line, texture, rendering, or a phrase like \"in the style "
    "of\". Style is attached later.\n"
    "Do not describe how anyone looks. Name only the people listed at the "
    "end of these rules, and name them exactly as written. A portrait of "
    "each of them is attached later. If that list is empty, the picture "
    "has no locked character.\n"
    "Do not repeat a moment already written in an earlier scene's prompt. "
    "When this scene is close to an earlier one and the texts are "
    "similar, a slight variation is not enough. Change the viewpoint, "
    "distance, arrangement, or what the picture emphasizes, so nearby "
    "pictures do not look alike.\n"
    "Where these rules say to keep the picture simple, drop what an image "
    "model cannot draw. They do not mean the prompt should be short. "
    "Describe the scene that remains in detail.\n"
    f"{RULES}"
)


def choose_iconic_scene(
    title: str | None,
    scenes: Sequence[str],
    characters: Sequence[Character],
    *,
    tier: Tier,
) -> str:
    """An iconic moment with many characters, drawn once as the style reference.

    The later cast and prompt steps are the same ones a scene uses. With no
    answer, the longest scene stands in for the moment.
    """
    result = ask(
        IconicScene,
        system=_ICONIC_SYSTEM,
        user=(
            f"{story_line(title)}"
            f"Character pool:\n{_pool_json(characters) if characters else '(none)'}\n\n"
            f"The storyboard:\n{cut(story_text(scenes)) or '(no text)'}\n\n"
            "After your analysis, describe the iconic scene."
        ),
        tier=tier,
    )
    if result is not None and result.scene.strip():
        return result.scene.strip()
    text = max((s.strip() for s in scenes if s and s.strip()), key=len, default="")
    if len(text) > 1200:
        text = text[:1200].rsplit(" ", 1)[0]
    return text or "The main characters of the story gathered in one place."


def context(scenes: Sequence[str], index: int, limit: int = CONTEXT_CHARS) -> str:
    """Scenes around `scenes[index]`, including it, up to `limit` characters."""
    if not scenes or index < 0 or index >= len(scenes):
        return ""
    start = end = index
    text = scenes[index]
    while True:
        grew = False
        if start > 0:
            piece = scenes[start - 1]
            if len(text) + len(piece) + 2 <= limit:
                start -= 1
                text = piece + "\n\n" + text
                grew = True
        if end + 1 < len(scenes):
            piece = scenes[end + 1]
            if len(text) + len(piece) + 2 <= limit:
                end += 1
                text = text + "\n\n" + piece
                grew = True
        if not grew:
            break
    return text


def _pool_json(characters: Sequence[Character]) -> str:
    rows = [{"id": person.id, "name": person.name, "role": person.role} for person in characters]
    return json.dumps(rows, ensure_ascii=False)


def _labels(person: Character) -> tuple[str, ...]:
    """Id and name, the two ways a pool character is named."""
    if person.name and person.name.casefold() != person.id.casefold():
        return (person.id, person.name)
    return (person.id,)


def _match_pool_id(ident: str, characters: Sequence[Character]) -> str:
    """Pool id for one model string.

    When exactly one pool character's name or id occurs in the string, that
    character is used. Otherwise the closest pool character by ratio.
    """
    people = [person for person in characters if person.id]
    if not people:
        return ""
    folded = ident.casefold()
    named = [
        person.id
        for person in people
        if any(label.casefold() in folded for label in _labels(person))
    ]
    if len(named) == 1:
        return named[0]
    best = people[0]
    best_score = -1.0
    for person in people:
        score = max(fuzz.ratio(folded, label.casefold()) for label in _labels(person))
        if score > best_score:
            best = person
            best_score = score
    return best.id


def _ids_in_pool(ids: Sequence[str], characters: Sequence[Character]) -> list[str]:
    """Every model id, mapped onto a pool character, in the model's order."""
    seen: set[str] = set()
    out: list[str] = []
    for ident in ids:
        matched = _match_pool_id(ident, characters)
        if matched and matched not in seen:
            seen.add(matched)
            out.append(matched)
    return out


def pick_cast(
    title: str | None,
    surrounding: str,
    scene: str,
    characters: Sequence[Character],
    *,
    tier: Tier,
) -> SceneCast:
    """Who from `characters` is visible in this scene."""
    if not characters:
        return SceneCast(analysis="No character pool; the picture has no locked figures.", character_ids=[])
    result = ask(
        SceneCast,
        system=_CAST_SYSTEM,
        user=(
            f"{story_line(title)}"
            f"Character pool (copy ids from this list):\n{_pool_json(characters)}\n\n"
            f"Surrounding scenes (this scene and its neighbours):\n{surrounding or '(none)'}\n\n"
            f"The scene to illustrate:\n{scene or '(empty)'}\n\n"
            "After your analysis, list the character ids visible in this scene."
        ),
        tier=tier,
    )
    if result is None:
        return SceneCast(analysis="No model answered; no characters selected.", character_ids=[])
    return SceneCast(
        analysis=result.analysis,
        character_ids=_ids_in_pool(
            [ident.strip() for ident in result.character_ids if ident and ident.strip()],
            characters,
        ),
    )


def _section(title: str, comment: str, body: str) -> str:
    """One part of the prompt-writer's input, with a comment the model can read.

    The heading names the part. Each line of `comment` says what the part
    is and how to use it. `body` is the part itself.
    """
    notes = "\n".join(f"# {line}" if line else "#" for line in comment.splitlines())
    return f"# {title}\n{notes}\n{body.strip() or '(none)'}"


def _prompt_line(number: int, prompt: str) -> str:
    """One earlier scene, as `<scene> - prompt` on a single line."""
    return f"{number} - {' '.join(prompt.split())}"


def _scene_rules(monochrome: bool) -> str:
    """Writing rules. The monochrome line is added only when asked."""
    if not monochrome:
        return _SCENE_RULES
    return f"{_SCENE_RULES}\n{MONOCHROME_NOTE}"


def _rules_section(chosen: Sequence[Character], *, monochrome: bool = False) -> str:
    cast = _pool_json(chosen) if chosen else "(none from the locked cast)"
    return _section(
        "Rules",
        "These rules govern the prompt you write. They are not a description "
        "of this scene, and they are not an artistic style. Follow them. "
        "The list at the end is who is physically visible in this picture: "
        "name only those people, exactly as written, and do not describe how "
        "they look.",
        f"{_scene_rules(monochrome)}\n\n"
        "Characters visible in this picture:\n"
        f"{cast}",
    )


def _earlier_section(earlier: Sequence[tuple[int, str]]) -> str:
    lines = [_prompt_line(number, prompt) for number, prompt in earlier if prompt and prompt.strip()]
    return _section(
        "Prompts already written",
        "Image prompts for earlier scenes of this storyboard, in order. "
        "Each line is `<scene> - prompt`: the scene number, a hyphen, "
        "then the prompt already written for that scene. These pictures "
        "already exist. Do not describe the same moment again. The scenes "
        "nearest this one matter most: if a nearby prompt shows almost the "
        "same situation, make this picture clearly different in viewpoint, "
        "distance, arrangement, or emphasis. A small change between close "
        "scenes looks like the same picture twice. Use the list only to "
        "see what has been shown, so this scene is a new picture. "
        "An empty list means no earlier scene has a prompt yet.",
        "\n".join(lines) if lines else "(none yet)",
    )


def _story_section(title: str | None, story: str) -> str:
    return _section(
        "The storyboard",
        "Every scene of the storyboard, in order. Read it so you know the "
        "story, who the people are, and what the scene below means in "
        "that story. Do not illustrate the whole storyboard. The picture is "
        "only the scene that follows this part.",
        f"{story_line(title)}{story.strip() or '(no text)'}",
    )


def _scene_section(scene: str, number: int | None) -> str:
    if number is None:
        comment = (
            "The moment to illustrate. It is not a numbered scene of the "
            "storyboard. Write the image prompt for this moment only."
        )
    else:
        comment = (
            f"Scene {number}, the only scene this picture shows. Write "
            "the image prompt for this scene. Do not illustrate a different "
            "part of the storyboard, and do not repeat an earlier scene's "
            "prompt. If a nearby scene shows almost the same situation, "
            "depict this one differently enough that the pictures do not "
            "look like slight variants."
        )
    return _section("This scene", comment, scene.strip() or "(empty)")


def _blocked_section(blocked: Sequence[dict]) -> str:
    body = "\n".join(f"- {b['prompt']}\n  refusal: {b['error']}" for b in blocked)
    return _section(
        "Prompts the image model refused",
        "These prompts were already sent for this same scene and refused, "
        "with the model's reply under each one. Tell the same moment "
        "another way. Do not repeat what got them refused.",
        body,
    )


def prompt_input(
    title: str | None,
    story: str,
    scene: str,
    chosen: Sequence[Character],
    *,
    scene_number: int | None = None,
    earlier: Sequence[tuple[int, str]] = (),
    blocked: Sequence[dict] = (),
    monochrome: bool = False,
) -> str:
    """The user message for one scene prompt.

    Order: rules, prompts already written (`<scene> - prompt`), the whole
    storyboard, then this scene. A refusal list is added only when a
    previous prompt for this scene was blocked. When `monochrome` is true,
    the rules require a description for a monochromatic image.
    """
    closing = (
        "After your analysis, write the prompt for the scene above. "
        "It is a detailed description of that scene for an image model: "
        "the scene itself, with no style clues and no description of "
        "how anyone looks."
    )
    if monochrome:
        closing = f"{closing} {MONOCHROME_NOTE}"
    parts = [
        _rules_section(chosen, monochrome=monochrome),
        _earlier_section(earlier),
        _story_section(title, story),
        _scene_section(scene, scene_number),
    ]
    if blocked:
        parts.append(_blocked_section(blocked))
    parts.append(_section("What to write", closing, "Write the prompt now."))
    return "\n\n".join(parts)


def write_prompt(
    title: str | None,
    story: str,
    scene: str,
    chosen: Sequence[Character],
    *,
    tier: Tier,
    scene_number: int | None = None,
    earlier: Sequence[tuple[int, str]] = (),
    blocked: Sequence[dict] = (),
    monochrome: bool = False,
) -> ScenePrompt:
    """The scene prompt the image model draws. Style and looks are added later.

    `story` is the whole storyboard. `scene` is the text to illustrate.
    `earlier` is `(scene number, prompt)` for every earlier scene that
    already has a prompt. `scene_number` is this scene, 1-based; the style
    reference passes none. When `monochrome` is true, the prompt is written
    for a monochromatic image and names no colours.
    """
    result = ask(
        scene_prompt_model(monochrome),
        system=_prompt_system(monochrome),
        user=prompt_input(
            title,
            story,
            scene,
            chosen,
            scene_number=scene_number,
            earlier=earlier,
            blocked=blocked,
            monochrome=monochrome,
        ),
        tier=tier,
    )
    if result is not None and result.prompt.strip():
        return result
    fallback = scene.strip() or "A moment from the story."
    if len(fallback) > 600:
        fallback = fallback[:600].rsplit(" ", 1)[0] + "."
    return ScenePrompt(analysis="No model answered; the scene text is used as the prompt.", prompt=fallback)


def style_reference_note() -> str:
    """The storyboard's style reference, always reference image 1."""
    return (
        "Attached reference image 1 is the style reference for this entire storyboard. "
        "Stick to the same artistic style: the same line, palette, medium, "
        "and rendering. Do not copy its subject, setting, or composition."
    )


def character_block(
    chosen: Sequence[Character],
    attached: Sequence[Character],
    *,
    image_offset: int = 0,
) -> str:
    """Who is in the picture, and which attached image is which person.

    `image_offset` is how many reference images come before the portraits,
    so "reference image N" still points at the right file.
    """
    if not chosen:
        return "Characters in this picture: none of the locked figures. Draw the scene without them."
    lines = [
        "Characters in this picture. Match each attached reference portrait "
        "to the person named here, and keep them looking like that portrait. "
        "Use the appearance paragraph if no portrait is attached."
    ]
    attached_ids = [person.id for person in attached]
    for i, person in enumerate(chosen, start=1):
        name = person.name or person.id
        appearance = person.appearance.strip() or "looks as in the attached portrait"
        if person.id in attached_ids:
            order = attached_ids.index(person.id) + 1 + image_offset
            ref = f"Attached reference image {order} is {name}."
        else:
            ref = f"No portrait is attached for {name}; follow the appearance paragraph."
        lines.append(f"{i}. {name}. {ref} {appearance}")
    return "\n".join(lines)


def scene_text(scene_prompt: str) -> str:
    """The scene itself. This is the last text in the image request."""
    return f"Scene to draw:\n{scene_prompt.strip()}"


def portraits_to_attach(chosen: Sequence[Character]) -> list[Character]:
    """Up to `MAX_REFERENCE_PORTRAITS` characters whose portrait file exists."""
    ready = [person for person in chosen if person.portrait is not None and person.portrait.is_file()]
    return ready[:MAX_REFERENCE_PORTRAITS]


def image_request(
    chosen: Sequence[Character],
    attached: Sequence[Character],
    scene_prompt: str,
    style_text: str,
    *,
    style_reference: bool,
) -> str:
    """Text sent with the reference images. The scene prompt is last.

    Order: style reference, cast, drawing rules, artistic style, then the
    scene. Reference image 1 is the style. Every later image is a character
    portrait. Nothing is appended after the scene.
    """
    parts: list[str] = []
    offset = 0
    if style_reference:
        parts.append(style_reference_note())
        offset += 1
    parts.append(character_block(chosen, attached, image_offset=offset))
    parts.append(RULES)
    if style_text.strip():
        parts.append(style_text.strip())
    parts.append(scene_text(scene_prompt))
    return "\n\n".join(parts)
