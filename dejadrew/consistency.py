"""Quality check for one picture.

Four separate requests, one criterion each, in this order. A criterion
runs only when the style sets its threshold. The picture is kept when
every set threshold is met. A score under its mark stops the later
checks. The caller draws the next seed, and after the attempt limit
scores every skipped criterion on every picture from that round, then
keeps the one with the best geometric mean.

    style        this picture against the storyboard's style reference;
                 no request, and no score, when that file does not exist yet
    characters   this picture against the portraits in the scene;
                 100, with no request, when the scene has no portraits
    technical    this picture alone: artefacts, an AI look, broken physics
    scene        this picture against the description it was drawn from

A request that returns nothing does not fail the picture, so a failing
model does not redraw forever.

`score_portrait` is separate. It scores one reference portrait against
the appearance paragraph it was drawn from.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

from .characters import Character
from .config import Tier
from .llm import ask
from .schemas import (
    ROLE,
    CharacterScore,
    PortraitScore,
    SceneScore,
    StyleScore,
    TechnicalScore,
)
from .style import Thresholds

_STRICT = (
    "Score 0-100. Be strict: reserve scores above 80 for a picture you "
    "would publish in this storyboard without hesitation.\n\n"
    "First, reason carefully about what you actually see, then score."
)

_STYLE_SYSTEM = (
    ROLE
    + "The style reference for the whole storyboard and one new illustration "
    "from it are attached. Your task is to score how closely the new "
    "scene matches the artistic style of the style reference.\n\n"
    "Judge only style: medium, linework, palette, texture, lighting, and "
    "how the picture is rendered. The subject, characters, setting, and "
    "composition are supposed to change. Do not lower the score because "
    "the scenes show different moments.\n\n"
    "100 means the same hand could have drawn both. A shift in medium, "
    "palette, or rendering drops the score hard.\n\n"
    + _STRICT
)

_STYLE_USER = (
    "Attached image 1 is the style reference for this entire storyboard. "
    "Attached image 2 is the new scene. After your analysis, score how "
    "similar their artistic styles are."
)

_CHARACTER_SYSTEM = (
    ROLE
    + "Character portraits are attached, and the new scene is the last "
    "image. Your task is to score how faithfully the scene draws those "
    "characters.\n\n"
    "Judge only identity: face, age, build, hair, skin, clothing, and "
    "distinctive marks, each person against their own portrait. Pose, "
    "action, expression, and setting are supposed to change. Do not lower "
    "the score for artistic style or for drawing defects. Those are scored "
    "separately.\n\n"
    "A character who is missing, or who could not be the person in their "
    "portrait, drops the score hard. Clothing may shift a little with the "
    "scene, but the person must still be recognisable as that portrait.\n\n"
    "100 means every attached character is unmistakably the same person as "
    "their portrait.\n\n"
    + _STRICT
)

_SCENE_SYSTEM = (
    ROLE
    + "A scene description and the illustration drawn for it are given. "
    "Your task is to score how faithfully the picture shows that scene.\n\n"
    "Judge only the depicted moment: who is there, what they are doing, and "
    "where it happens. A readable simplification is still the same moment. "
    "A different event, the wrong place, or a missing main action drops the "
    "score hard. Do not lower the score for artistic style, for whether a "
    "face matches a reference portrait, or for drawing defects. Those are "
    "scored separately.\n\n"
    "100 means a reader of the description would recognise this exact moment.\n\n"
    + _STRICT
)

_TECHNICAL_SYSTEM = (
    ROLE
    + "One illustration is attached. Your task is to score its technical "
    "accuracy.\n\n"
    "Judge only defects a viewer would notice. Reason in this order. "
    "Artefacts: extra or missing fingers, melted or fused hands, extra or "
    "missing limbs, wrong facial features, bodies merging into each other "
    "or into objects, smeared detail, tiling, seams, and any text, letters, "
    "or pseudo-writing. AI look: plastic or airbrushed skin, glossy "
    "over-rendered surfaces, generic symmetric faces, glow, haze everywhere, "
    "meaningless ornament, uniform polish with no visible hand or medium. "
    "Physics: objects that float or intersect, supports that cannot hold "
    "the weight, light and shadows that disagree, scale that does not work, "
    "cloth, hair, water, or smoke that ignores gravity.\n\n"
    "A hand-drawn or sketchy medium is not a defect. Rough pencil and open "
    "lines are fine when they are the style. Score the mistakes, not the "
    "medium.\n\n"
    "100 means nothing is wrong even under scrutiny. One obvious defect "
    "drops the score below 50.\n\n"
    + _STRICT
)

_TECHNICAL_USER = "Attached: the illustration. After your analysis, score its technical accuracy."

_PORTRAIT_SYSTEM = (
    ROLE
    + "A written appearance and the reference portrait drawn from it are "
    "given. Your task is to score how faithfully the portrait shows that "
    "person.\n\n"
    "Judge only the big facts the description actually names: age, build, "
    "colours, hair or coat, the clothes it lists, and a mark it treats as "
    "theirs. A neutral standing pose and a plain background are required "
    "of this sheet and are not part of the score. Do not lower the score "
    "for artistic style or for drawing defects. Those are scored "
    "separately.\n\n"
    "A named fact that is missing, or drawn as something else — a white "
    "horse drawn brown, a red cloak left out — drops the score hard. Do "
    "not lower the score for a detail the paragraph never mentioned, or "
    "for a small feature that can change from picture to picture.\n\n"
    "100 means the portrait is unmistakably the person the paragraph describes.\n\n"
    + _STRICT
)

_Scored = TypeVar("_Scored", StyleScore, CharacterScore, PortraitScore, SceneScore, TechnicalScore)

# A miss stops the rest. After the attempt limit the caller fills these in.
_ORDER = ("style", "characters", "technical", "scene")


@dataclass(frozen=True)
class Score:
    """One criterion. `value` is None when it was not scored."""

    value: int | None
    reason: str = ""
    analysis: str = ""


@dataclass(frozen=True)
class Report:
    style: Score
    characters: Score
    scene: Score
    technical: Score

    def accepted(self, thresholds: Thresholds) -> bool:
        """True when every set threshold is met.

        A criterion with no score (no answer, no style reference, no scene
        text) does not fail the picture.
        """
        for item in fields(thresholds):
            need = getattr(thresholds, item.name)
            got = getattr(self, item.name).value
            if need is None or got is None:
                continue
            if got < need:
                return False
        return True

    def geometric_mean(self) -> float | None:
        """Geometric mean of the scores that were given.

        None when nothing was scored. A 0 on any criterion makes the mean 0.
        """
        values: list[int] = []
        for item in fields(self):
            score: Score = getattr(self, item.name)
            if score.value is not None:
                values.append(score.value)
        if not values:
            return None
        if any(value <= 0 for value in values):
            return 0.0
        return math.prod(values) ** (1.0 / len(values))

    def pending(self, thresholds: Thresholds) -> bool:
        """True when a set criterion was skipped after an earlier miss."""
        for name in _ORDER:
            need = getattr(thresholds, name)
            if need is None:
                continue
            score: Score = getattr(self, name)
            if score.value is None and not score.reason:
                return True
        return False


def check(
    image: Path,
    *,
    thresholds: Thresholds,
    tier: Tier,
    style_reference: Path | None = None,
    characters: Sequence[Character] = (),
    scene: str = "",
    on_score: Callable[[str, Score], None] | None = None,
    stop_on_fail: bool = True,
    prior: Report | None = None,
) -> Report:
    """Score `image` on every criterion that has a threshold.

    Order is style, characters, technical, scene. `characters` are the
    people in this scene. With no portrait files the character score is
    100 and no request is made. `scene` is the description the picture
    was drawn from. `on_score` is called with the criterion name and its
    score as soon as that criterion is done.

    `stop_on_fail` leaves every later criterion unscored once a score is
    under its mark. `prior` keeps scores already obtained for this same
    picture, so a later pass pays only for the ones that were skipped.
    """
    scores = {name: Score(None) for name in _ORDER}
    for name in _ORDER:
        need = getattr(thresholds, name)
        if need is None:
            continue
        existing = _kept(prior, name)
        if existing is not None:
            scores[name] = existing
            if stop_on_fail and _missed(existing, need):
                break
            continue
        score = _criterion(
            name,
            image,
            tier,
            style_reference=style_reference,
            characters=characters,
            scene=scene,
        )
        scores[name] = score
        if on_score is not None:
            on_score(name, score)
        if stop_on_fail and _missed(score, need):
            break
    return Report(
        style=scores["style"],
        characters=scores["characters"],
        scene=scores["scene"],
        technical=scores["technical"],
    )


def score_portrait(image: Path, character: Character, *, tier: Tier) -> Score:
    """How faithfully `image` shows `character.appearance`.

    No request when the appearance or the file is missing. A request that
    returns nothing does not fail the portrait.
    """
    appearance = character.appearance.strip()
    if not appearance:
        return Score(None, "No appearance description.")
    if not image.is_file():
        return Score(None, "No portrait.")
    name = character.name or character.id
    user = (
        f"Character: {name}.\n\n"
        f"Appearance to match:\n{appearance}\n\n"
        "Attached: the reference portrait drawn for that appearance. After "
        "your analysis, score how faithfully it shows that person."
    )
    return _ask(PortraitScore, _PORTRAIT_SYSTEM, user, [image], tier)


def report_scores(report: Report) -> dict[str, dict]:
    """The criteria that have a value, as `{name: {score, reason, analysis}}`."""
    out: dict[str, dict] = {}
    for item in fields(report):
        score: Score = getattr(report, item.name)
        if score.value is not None:
            out[item.name] = as_dict(score)
    return out


def as_dict(score: Score) -> dict:
    return {"score": score.value, "reason": score.reason, "analysis": score.analysis}


def from_dict(raw: Mapping | None) -> Score | None:
    """A score stored by `as_dict`, or None when there is no usable value."""
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("score")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return Score(value, str(raw.get("reason") or ""), str(raw.get("analysis") or ""))


def summary(report: Report, thresholds: Thresholds, *, ok: bool, reasons: bool = True) -> str:
    """One line of `name score/threshold`, in check order, plus a reason for each miss."""
    parts: list[str] = []
    notes: list[str] = []
    for name in _ORDER:
        need = getattr(thresholds, name)
        if need is None:
            continue
        score: Score = getattr(report, name)
        shown = "-" if score.value is None else str(score.value)
        parts.append(f"{name} {shown}/{need}")
        if reasons and score.reason and (score.value is None or score.value < need):
            notes.append(f"  {name}: {score.reason}")
    line = ", ".join(parts)
    line += " - keep" if ok else " - redraw"
    if notes:
        line += "\n" + "\n".join(notes)
    return line


def _kept(prior: Report | None, name: str) -> Score | None:
    """A score already decided for this picture, or None when it still has to run."""
    if prior is None:
        return None
    score: Score = getattr(prior, name)
    if score.value is not None or score.reason:
        return score
    return None


def _missed(score: Score, need: int) -> bool:
    return score.value is not None and score.value < need


def _criterion(
    name: str,
    image: Path,
    tier: Tier,
    *,
    style_reference: Path | None,
    characters: Sequence[Character],
    scene: str,
) -> Score:
    if name == "style":
        return _style(style_reference, image, tier)
    if name == "characters":
        return _characters(characters, image, tier)
    if name == "technical":
        return score_technical(image, tier=tier)
    return score_scene(scene, image, tier=tier)


def _style(reference: Path | None, image: Path, tier: Tier) -> Score:
    if reference is None or not reference.is_file():
        return Score(None, "No style reference.")
    return _ask(StyleScore, _STYLE_SYSTEM, _STYLE_USER, [reference, image], tier)


def _characters(people: Sequence[Character], image: Path, tier: Tier) -> Score:
    portraits: list[tuple[Character, Path]] = []
    for person in people:
        portrait = person.portrait
        if portrait is not None and portrait.is_file():
            portraits.append((person, portrait))
    if not portraits:
        return Score(100, "No characters in this picture.")
    lines = [
        f"Attached image {index} is the portrait of {person.name or person.id}."
        for index, (person, _portrait) in enumerate(portraits, start=1)
    ]
    lines.append(
        f"Attached image {len(portraits) + 1} is the new scene. After your analysis, "
        "score how correctly those characters are drawn in it."
    )
    paths = [portrait for _person, portrait in portraits]
    paths.append(image)
    return _ask(CharacterScore, _CHARACTER_SYSTEM, "\n".join(lines), paths, tier)


def score_scene(scene: str, image: Path, *, tier: Tier) -> Score:
    """How faithfully `image` shows `scene`. No request when `scene` is empty."""
    text = scene.strip()
    if not text:
        return Score(None, "No scene description.")
    user = (
        f"Scene description:\n{text}\n\n"
        "Attached: the illustration drawn for that description. After your "
        "analysis, score how faithfully it shows the scene."
    )
    return _ask(SceneScore, _SCENE_SYSTEM, user, [image], tier)


def score_technical(image: Path, *, tier: Tier) -> Score:
    """Artefacts, an AI look, and broken physics in `image` alone."""
    return _ask(TechnicalScore, _TECHNICAL_SYSTEM, _TECHNICAL_USER, [image], tier)


def _ask(output: type[_Scored], system: str, user: str, images: Sequence[Path], tier: Tier) -> Score:
    verdict = ask(output, system=system, user=user, tier=tier, images=images)
    if verdict is None:
        return Score(None, "No model answered.")
    return Score(verdict.score, verdict.reason.strip(), verdict.analysis.strip())
