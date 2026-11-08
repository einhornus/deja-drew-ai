"""Draw one character portrait or one scene picture.

Portraits are a generic standing figure, saved at `characters/<id>.png`
so a later scene can attach them. A scene picture is cropped to the
style's width and height. Its reference images start with the
storyboard's style reference, then the character portraits. The scene
prompt is the last text in the request.

A refused prompt is rewritten and retried on the same model, up to
`MAX_REWRITES` times. Auth and rate-limit errors are raised so the caller
can try the picture again on the next run. Every picture is drawn at the
seed the caller passes, and a seeded image is cached, so the same seed
comes back as the same unnormalized image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageOps

from . import imagegen
from .characters import Character
from .config import Tier
from .scenes import image_request, portraits_to_attach, write_prompt
from .style import Style

MAX_REWRITES = 3

_PORTRAIT_RULES = (
    "Draw a generic appearance sheet of this one character: a single "
    "standing figure, three-quarter or full body, face clearly visible, "
    "neutral pose, plain uncluttered background. This is their typical "
    "look, not a scene from the story. Follow the big facts in the "
    "appearance — colours, build, hair or coat, and the clothes named. "
    "Do not invent a different colour or costume, and do not add marks "
    "the appearance does not mention. "
    "No other people.\n"
    "No text of any kind anywhere in the image: no letters, words, numbers, "
    "titles, captions, signatures, monograms, logos, or watermarks."
)


def save_png(data: bytes, path: Path, size: tuple[int, int]) -> None:
    img = imagegen.decode(data)
    fitted = ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(path, "PNG")


def _style_text(style: Style) -> str:
    return f"Artistic style:\n{style.prompt}"


def _draw(
    style: Style,
    prompt: str,
    path: Path,
    reference: Sequence[Path] | None,
    seed: int | None = None,
    *,
    include_style: bool = True,
) -> None:
    """Draw `prompt`. Portraits append the style after it.

    A scene already ends with its own scene sentence, so it passes
    `include_style=False` and carries the style paragraph inside `prompt`,
    just before that sentence.
    """
    data = imagegen.generate(
        style.model,
        prompt,
        reference=reference or None,
        aspect_ratio=style.aspect_ratio,
        seed=seed,
        style=_style_text(style) if include_style else None,
        style_last=True,
        quality=style.quality,
    )
    save_png(data, path, (style.width, style.height))


def _portrait_prompt(character: Character, *, safer: bool) -> str:
    name = character.name or character.id
    appearance = character.appearance.strip() or name
    prompt = f"Character to draw: {name}.\nAppearance:\n{appearance}\n\n{_PORTRAIT_RULES}"
    if safer:
        prompt += (
            "\nKeep the picture suitable for a general-audience storyboard; "
            "no violence, no sexual content."
        )
    return prompt


def draw_character(style: Style, character: Character, path: Path, *, seed: int = 1) -> None:
    """One generic portrait at `seed`. Raises ImageGenError on failure.

    Normalization, when the style defines it, happens after this returns.
    """
    prompt = _portrait_prompt(character, safer=False)
    for attempt in range(MAX_REWRITES + 1):
        try:
            _draw(style, prompt, path, None, seed)
            return
        except imagegen.ImageGenError as exc:
            if not exc.blocked or attempt == MAX_REWRITES:
                raise
            prompt = _portrait_prompt(character, safer=True)


def draw_scene(
    title: str | None,
    story: str,
    scene: str,
    chosen: Sequence[Character],
    output: Path,
    style: Style,
    planned: str,
    *,
    tier: Tier,
    style_reference: Path | None = None,
    seed: int | None = None,
    scene_number: int | None = None,
    earlier: Sequence[tuple[int, str]] = (),
) -> tuple[str, list[dict]]:
    """Draw one scene into `output`. Returns the prompt that was drawn and any refusals.

    `style_reference`, when that file exists, is reference image 1. The
    character portraits follow it. The scene prompt is the last text in
    the request, after the cast, the rules, and the artistic style.
    `story`, `scene`, `scene_number`, and `earlier` are only used when a
    prompt is refused and has to be rewritten. `seed=None` always draws a
    new picture. An int reuses the cached image for this model, prompt,
    references, and seed.
    """
    blocked: list[dict] = []
    prompt = planned
    style_ok = style_reference is not None and style_reference.is_file()
    for attempt in range(MAX_REWRITES + 1):
        attached = portraits_to_attach(chosen)
        refs: list[Path] = []
        if style_ok and style_reference is not None:
            refs.append(style_reference)
        refs.extend(person.portrait for person in attached if person.portrait is not None)
        body = image_request(chosen, attached, prompt, _style_text(style), style_reference=style_ok)
        try:
            _draw(style, body, output, refs, seed, include_style=False)
            return prompt, blocked
        except imagegen.ImageGenError as exc:
            if not exc.blocked or attempt == MAX_REWRITES:
                raise
            blocked.append({"prompt": prompt, "error": str(exc)})
            prompt = write_prompt(
                title,
                story,
                scene,
                chosen,
                tier=tier,
                scene_number=scene_number,
                earlier=earlier,
                blocked=blocked,
                monochrome=style.monochrome,
            ).prompt.strip()
    return prompt, blocked
