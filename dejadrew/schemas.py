"""LLM-typed shapes.

`analysis` is always declared first. Constrained decoding emits keys in
declaration order, so the model has to reason about the agenda in that
field before it is allowed to commit to a cast, an appearance, a prompt,
or a score.
"""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field, create_model

# Opens every system prompt.
ROLE = (
    "You're the art director of an illustrated storyboard. The storyboard "
    "is a list of scenes, and an illustrator draws one picture per scene. "
)

# Appended to a character or scene description prompt, and to the fields
# that hold that description, only when the style's `monochrome` flag is set.
MONOCHROME_NOTE = (
    "The description must be for a monochromatic image. "
    "Do not name colours or hues; describe shape, light and dark, and texture only."
)


class CharacterRef(BaseModel):
    id: str = Field(
        description=(
            "Stable English slug, lowercase letters, digits, and underscores "
            "only, e.g. 'the_emperor'. Unique in this list. Copied later to "
            "pick who is in a scene"
        )
    )
    name: str = Field(
        description="How the storyboard names them, in the language of the text, e.g. 'the Emperor'"
    )
    role: str = Field(
        description="Who they are in the story, in English, a few words, e.g. 'vain ruler of the city'"
    )


class CharacterList(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before any character is listed. Cover, in "
            "order: who recurs as a distinct figure a viewer would recognise "
            "from one picture to the next; who is named versus known only by "
            "a role; which animals, spirits, or personified things have a "
            "stable look of their own; who is a one-off extra, a crowd, or "
            "an offstage mention and so must be left out; how to slug each "
            "kept name so later calls can copy the id exactly."
        )
    )
    characters: list[CharacterRef] = Field(
        description=(
            "Recurring figures worth drawing consistently, main characters "
            "first. Skip unnamed crowds, one-scene extras, and anyone who "
            "never appears in a picture"
        )
    )


class CharacterAppearance(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the appearance is written. Cover, in "
            "order: which facts a viewer would notice at a glance and "
            "which of those stay the same from picture to picture (broad "
            "age, build if it shows, hair or coat colour, the few usual "
            "clothes and their colours, a mark that identifies them, such "
            "as the colour of a horse); which of those the text states and "
            "which to infer in one phrase from role, rank, era, and "
            "country; which details to omit because they are small or "
            "change with time (exact age, facial features, fabric, cut, "
            "footwear, dirt, injuries, expression). Keep only what makes "
            "the drawing easier."
        )
    )
    appearance: str = Field(
        description=(
            "A few plain sentences, in English, easy for an image model "
            "to draw. Only the big, stable, eye-catching facts: broad "
            "age, build if it shows, hair or coat colour and its simple "
            "shape, the few usual clothes and their colours, and a mark "
            "only when it identifies the character. No inventory of the "
            "face, no fabrics or cuts, no small details, and nothing "
            "that changes from scene to scene. Appearance only — no "
            "pose, setting, action, or artistic style"
        )
    )


class IconicScene(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the scene is chosen. Cover, in "
            "order: which moments a viewer would recognise as this story's "
            "images; which of those put the most characters from the pool "
            "in one place, visible at the same time; which setting and "
            "action an image model can draw clearly; the one moment to use "
            "as the style reference for every picture in the storyboard."
        )
    )
    scene: str = Field(
        description=(
            "That iconic moment, as a short passage of what is happening "
            "and who is there, naming characters as the pool names them. "
            "As many pool characters as the moment can honestly hold, all "
            "of them physically present and visible. Do not describe "
            "clothing, artistic style, or how to draw the picture — a "
            "later step writes the image prompt"
        )
    )


class SceneCast(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before any id is listed. Cover, in order: "
            "what is happening in this scene; who of the given pool is "
            "physically present and would be visible in a picture of this "
            "moment; who is only mentioned, offstage, or in the surrounding "
            "scenes and so must be left out; whether the picture should "
            "be a setting or object with no one from the pool (then the "
            "list is empty); the order, most important / most on-camera first."
        )
    )
    character_ids: list[str] = Field(
        description=(
            "Ids copied exactly from the pool of who is visible in this "
            "scene, most on-camera first. Empty if the picture has no one "
            "from the pool"
        )
    )


class ScenePrompt(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the prompt is written. Cover, in "
            "order: what this scene shows, and where that moment sits in "
            "the story; which earlier prompts already show a nearby moment, "
            "and how this picture stays a different moment — when those "
            "scenes are close and the texts are similar, by a real "
            "change of viewpoint, distance, arrangement, or emphasis, not "
            "a slight variation; who is visible "
            "and what each of them is doing, by the name given, not by a "
            "new nickname; the place, the objects that matter, and how the "
            "figures are arranged; which parts an image model would botch "
            "(crowds, hands, distant faces, machinery, reflections, writing, "
            "intricate patterns) and how to cut them; confirmation that the "
            "prompt will carry no artistic style, no medium, no palette, no "
            "line, no texture, no rendering, no description of how anyone "
            "looks, and no text."
        )
    )
    prompt: str = Field(
        description=(
            "A detailed prompt for an image model, in English, of this "
            "scene only: who is there (by the names given), what each "
            "person is doing, where they are, the objects that matter, and "
            "how the moment is arranged. Concrete and visual, as long as "
            "the moment needs and no longer. No artistic style, medium, "
            "palette, line, texture, rendering, or 'in the style of'. Do "
            "not describe how anyone looks. Do not mention text or titles"
        )
    )


def _monochrome_variant(model: type[BaseModel], fields: tuple[str, ...]) -> type[BaseModel]:
    """Same shape, with `MONOCHROME_NOTE` added to the named field descriptions."""
    note = f" {MONOCHROME_NOTE}"
    overrides = {}
    for name in fields:
        info = model.model_fields[name]
        overrides[name] = (
            info.annotation,
            Field(description=f"{info.description or ''}{note}"),
        )
    return create_model(f"{model.__name__}Monochrome", __base__=model, **overrides)


_APPEARANCE_MONOCHROME = cast(
    type[CharacterAppearance],
    _monochrome_variant(CharacterAppearance, ("analysis", "appearance")),
)
_SCENE_PROMPT_MONOCHROME = cast(
    type[ScenePrompt],
    _monochrome_variant(ScenePrompt, ("analysis", "prompt")),
)


def character_appearance_model(monochrome: bool) -> type[CharacterAppearance]:
    """Appearance schema. The monochrome variant forbids colour names."""
    if monochrome:
        return _APPEARANCE_MONOCHROME
    return CharacterAppearance


def scene_prompt_model(monochrome: bool) -> type[ScenePrompt]:
    """Scene-prompt schema. The monochrome variant forbids colour names."""
    if monochrome:
        return _SCENE_PROMPT_MONOCHROME
    return ScenePrompt


class StyleScore(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the score. Cover, in order: the "
            "medium, linework, and edges of the style reference; its palette "
            "and how colour is applied; its surface texture and lighting; "
            "the same four things in the new scene; where the new scene is "
            "the same hand and where it drifts (a different medium, a "
            "smoother or more photographic render, a new palette, different "
            "line weight). Ignore subject, characters, setting, and "
            "composition — those are supposed to change."
        )
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Style consistency of the new scene with the storyboard's style "
            "reference, 0-100. 100 = the same medium, line, palette, texture, "
            "and rendering; 0 = a different style entirely. Subject matter "
            "does not count"
        ),
    )
    reason: str = Field(
        description="One sentence on whether the new scene matches the style reference"
    )


class CharacterScore(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the score. Cover, in order: each "
            "attached portrait (face, age, build, hair, skin, clothing, "
            "distinctive marks); where that person appears in the scene, if "
            "they appear; what matches the portrait and what makes them look "
            "like someone else; any portrait that is missing from the scene. "
            "Ignore pose, action, expression, setting, artistic style, and "
            "drawing defects — pose and setting are supposed to change, and "
            "the other qualities are scored separately."
        )
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Character consistency, 0-100. 100 = every attached character is "
            "unmistakably the person in their portrait; 0 = none of them are. "
            "A missing character or a different person drops the score hard"
        ),
    )
    reason: str = Field(
        description="One sentence on whether the characters match their portraits"
    )


class PortraitScore(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the score. Cover, in order: the big "
            "facts the appearance actually names (broad age, build, colours, "
            "hair or coat, the clothes listed, an identifying mark); whether "
            "the portrait shows each of those; any named fact that is "
            "missing or drawn as something else. Do not look for features "
            "the paragraph left out. Ignore the standing pose, the plain "
            "background, artistic style, and drawing defects."
        )
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Character consistency of this reference portrait with the "
            "written appearance, 0-100. 100 = unmistakably the person "
            "described; 0 = a different person. A named feature that is "
            "missing or contradicted drops the score hard"
        ),
    )
    reason: str = Field(
        description="One sentence on whether the portrait matches the written appearance"
    )


class SceneScore(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the score. Cover, in order: what the "
            "description says is happening, who is there, and where; what "
            "the picture actually shows; which of those match; what is "
            "missing or contradicted (a different event, the wrong place, "
            "the main action absent). A plain simplification can still be "
            "the same moment. Ignore artistic style, whether a face matches "
            "some other reference, and drawing defects."
        )
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Scene fidelity, 0-100. 100 = a reader of the description would "
            "recognise this exact moment; 0 = a different event entirely"
        ),
    )
    reason: str = Field(
        description="One sentence on whether the picture shows the described scene"
    )


class TechnicalScore(BaseModel):
    analysis: str = Field(
        description=(
            "Directed reasoning before the score. Cover, in order: artefacts "
            "(fingers, hands, limbs, faces, bodies fused with each other or "
            "with objects, smeared detail, seams, any text or pseudo-writing); "
            "an AI look (plastic skin, glossy surfaces, generic symmetry, "
            "glow, meaningless ornament, no visible medium); physics "
            "(floating or intersecting objects, impossible supports, light "
            "and shadow that disagree, broken scale, cloth, hair, or water "
            "that ignores gravity). A sketchy or hand-drawn medium is not "
            "itself a defect."
        )
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Technical accuracy, 0-100. 100 = nothing wrong under scrutiny; "
            "one obvious artefact, AI tell, or physics error drops it below 50"
        ),
    )
    reason: str = Field(
        description="One sentence on artefacts, AI look, and physics in this picture"
    )
