"""One consistent picture per storyboard scene, in a style from a JSON file.

`draw` writes pictures and nothing else:

    characters/<id>.png     that character's generic look
    style.png               one iconic scene, the style reference for the storyboard
    1.png, 2.png, ...       one picture per scene

Pipeline
    1. Ask for the cast, then one appearance paragraph each.
    2. Draw any missing portrait, then normalize it. When
       `thresholds.characters` is set, score the portrait against its
       appearance. The bar is stricter than that mark T,
       `sqrt(T/100)*100`, and the portrait is redrawn at the next seed
       until it passes, at most three times `attempts`. If none pass,
       the highest character score is kept. With no character mark,
       the portrait is drawn once at seed 1.
    3. Draw the style reference if it is missing: an iconic scene with
       many characters, cast and prompted like any scene, then normalize
       it. With any quality threshold it is redrawn like a scene. There
       is no style picture yet, so the style criterion is not scored.
    4. For each scene whose picture is missing or out of date: who is
       visible, then the scene prompt. The prompt call sees the rules,
       every earlier scene's prompt, the whole storyboard, and this scene,
       and describes the scene only. Then draw, then normalize. The style
       reference is reference image 1, the portraits follow it. When the
       style sets any quality threshold, draw again at the next seed until
       the picture passes every set criterion, at most `attempts` times.
       Checks run in order: style, characters, technical, scene, and stop
       at the first miss. If none pass, score every criterion on every
       attempt and keep the picture with the best geometric mean.

When the style sets `monochrome` to true, each appearance and each scene
prompt is written for a monochromatic image and does not name colours.

The prompt each picture was drawn from is stored in the PNG, and so are
the quality scores of the kept picture. Read them back with `marks`.

A picture already on disk is kept. Delete it to draw it again. A scene
picture is also drawn again when its scene text changed: the hash of that
text is stored in the PNG. The LLM cache returns the previous answer when
a question has not changed, and the image cache does the same for a
repeated seed, so a rerun after an interruption is cheap.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import characters, config, consistency, draw as drawing, normalize, scenes
from .characters import Character
from .config import Tier
from .style import Style, load as load_style, portrait_attempts, portrait_threshold

log = logging.getLogger("dejadrew")

# Marks kept inside the PNG. "scene" is the hash of the scene text the
# picture was drawn for. "normalized" is "1" after the style's curve has
# been applied, so a rerun does not apply it again. "prompt" is the image
# prompt that was drawn, so a later scene can see it without asking again.
# "scores" is the JSON of the kept picture's quality scores, when it was checked.
_SCENE = "scene"
_NORMALIZED = "normalized"
_PROMPT = "prompt"
_SCORES = "scores"
_STYLE_REFERENCE = "style.png"


def draw(
    storyboard: Sequence[str],
    style: str | Path | dict | Style,
    output: str | Path,
    *,
    title: str | None = None,
    api_key: str | None = None,
    tier: Tier = "high",
    seed: int | None = None,
    models: str | Sequence[str] | None = None,
    tier_models: Mapping[str, str | Sequence[str]] | None = None,
    cache_dir: str | Path | None = None,
    cache: bool | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Draw one picture per string of `storyboard` into `output`.

    `storyboard` is a list of scene descriptions, in order. Each becomes
    `<n>.png`, numbered from 1. `style` is a path to a style JSON file,
    the name of a bundled style (`black_and_white`), or the parsed dict.
    `title` is optional context for every LLM call.

    `api_key` is your OpenRouter key. `tier` (`high`, `medium`, `low`) picks
    the LLM models: that tier's list, then every cheaper one. `tier_models`
    replaces the lists of the tiers it names, e.g. `{"high": [...]}`.
    `models` replaces the whole chain with your own OpenRouter ids, tried
    in order. `cache_dir` and `cache` control the answer and image caches,
    and `max_tokens` the output budget of one LLM call. Every one of these
    left as None is read from the environment or `.env` (see `config`).

    Scene pictures use `seed` when the style sets no quality threshold:
    no seed draws a new picture, and an int reuses the cached image of
    that model, prompt, and seed. Any threshold starts each picture at
    seed 1 and walks seeds up to the style's `attempts`.

    Returns a JSON-able summary. `ok` is false when any picture failed;
    the error of each is under `failed`, `portrait_failed`, and
    `style_failed`. Run again to retry only what is missing.
    """
    if isinstance(storyboard, str):
        raise TypeError("storyboard must be a sequence of strings, not a single string")
    texts = [s if isinstance(s, str) else str(s) for s in storyboard]
    if not texts or not any(t.strip() for t in texts):
        raise ValueError("storyboard is empty")
    settings = config.resolve(
        api_key=api_key,
        models=models,
        tier_models=tier_models,
        cache_dir=cache_dir,
        cache=cache,
        max_tokens=max_tokens,
    )
    if not settings.api_key:
        raise RuntimeError("No OpenRouter key. Pass api_key=, set OPENROUTER_API_KEY, or put it in .env.")
    spec = load_style(style)
    folder = Path(output).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    _clear_partials(folder)

    with config.use(settings):
        log.info("Casting %d scenes", len(texts))
        people = characters.build(title, texts, tier=tier, monochrome=spec.monochrome)
        log.info("Cast: %s", ", ".join(p.id for p in people) or "(none)")
        portraits_drawn, portraits_normalized, portrait_failed = _draw_portraits(folder, people, spec, tier=tier)
        style_drawn, style_normalized, style_failed = _draw_style_reference(
            folder, title, texts, people, spec, tier=tier
        )
        illustrated, normalized, failed = _draw_scenes(
            folder,
            title,
            texts,
            people,
            spec,
            tier=tier,
            seed=seed,
            style_reference=folder / _STYLE_REFERENCE,
        )

    wrote = [f"characters/{person.id}.png" for person in people if _exists(person.portrait)]
    if (folder / _STYLE_REFERENCE).is_file():
        wrote.append(_STYLE_REFERENCE)
    wrote += [f"{n}.png" for n in range(1, len(texts) + 1) if (folder / f"{n}.png").is_file()]
    return {
        "ok": not failed and not style_failed and not portrait_failed,
        "folder": str(folder),
        "title": title,
        "style": spec.name,
        "image_model": spec.model,
        "size": [spec.width, spec.height],
        "llm": settings.llm_models(tier)[0],
        "attempts": spec.attempts,
        "thresholds": {
            "style": spec.thresholds.style,
            "characters": spec.thresholds.characters,
            "scene": spec.thresholds.scene,
            "technical": spec.thresholds.technical,
        },
        "normalize": spec.normalize is not None,
        "characters": [
            {"id": p.id, "name": p.name, "role": p.role, "appearance": p.appearance} for p in people
        ],
        "scenes": len(texts),
        "portraits_drawn": portraits_drawn,
        "style_drawn": style_drawn,
        "illustrated": illustrated,
        "reused": len(texts) - illustrated - len(failed),
        "normalized": portraits_normalized + style_normalized + normalized,
        "failed": failed,
        "portrait_failed": portrait_failed,
        "style_failed": style_failed,
        "wrote": wrote,
    }


def _draw_portraits(folder: Path, people: list[Character], style: Style, *, tier: Tier) -> tuple[int, int, list[dict]]:
    """Draw each missing portrait, then normalize it. Returns drawn, normalized, failures.

    When `thresholds.characters` is set, each new portrait is scored
    against its appearance and redrawn until it passes or the attempt
    budget runs out. A portrait already on disk is kept, then normalized
    once if it has never been.
    """
    drawn = 0
    normalized = 0
    failed: list[dict] = []
    directory = folder / "characters"
    directory.mkdir(parents=True, exist_ok=True)
    for person in people:
        path = directory / f"{person.id}.png"
        person.portrait = path
        try:
            if not path.is_file():
                log.info("Drawing portrait %s", person.id)
                if _draw_one_portrait(person, path, style, tier=tier):
                    normalized += 1
                drawn += 1
            elif _normalize_once(path, style):
                normalized += 1
        except Exception as exc:
            log.error("character %s failed: %s", person.id, _error(exc))
            failed.append({"id": person.id, "error": _error(exc)})
    return drawn, normalized, failed


def _draw_one_portrait(person: Character, path: Path, style: Style, *, tier: Tier) -> bool:
    """Draw `path` for `person`. Returns whether the kept file was normalized.

    With no character threshold, one picture at seed 1. With a character
    threshold, seeds run from 1 up to three times `attempts`. Each picture
    is scored against the appearance. The first that meets the raised bar
    is kept. If none do, the highest score is kept.
    """
    mark = style.thresholds.characters
    if mark is None:
        drawing.draw_character(style, person, path, seed=1)
        return _normalize_once(path, style)
    tmp = path.with_name(f".{path.stem}.partial.png")
    try:
        _portrait_until_accepted(
            person,
            tmp,
            style,
            tier=tier,
            bar=portrait_threshold(mark),
            attempts=portrait_attempts(style.attempts),
            label=f"character {person.id}",
        )
        normalized = _tags(tmp).get(_NORMALIZED) == "1"
        tmp.replace(path)
        return normalized
    finally:
        tmp.unlink(missing_ok=True)


def _portrait_until_accepted(
    person: Character,
    tmp: Path,
    style: Style,
    *,
    tier: Tier,
    bar: int,
    attempts: int,
    label: str,
) -> None:
    """Draw seeds 1..`attempts` into `tmp`. Keep the first pass, else the best score.

    A portrait the model cannot score does not fail, so a failing model
    does not redraw forever. The picture left in `tmp` is the one to publish.
    """
    drawn: list[tuple[int, bytes, consistency.Score]] = []
    for seed in range(1, attempts + 1):
        drawing.draw_character(style, person, tmp, seed=seed)
        _normalize_once(tmp, style)
        score = consistency.score_portrait(tmp, person, tier=tier)
        ok = score.value is None or score.value >= bar
        shown = "-" if score.value is None else str(score.value)
        line = f"{label} seed {seed}: characters {shown}/{bar} - {'keep' if ok else 'redraw'}"
        if score.reason:
            line += f" ({score.reason})"
        log.info(line)
        if ok:
            return
        drawn.append((seed, tmp.read_bytes(), score))
    best: tuple[int, int, bytes] | None = None
    for seed, image, score in drawn:
        value = score.value if score.value is not None else -1
        if best is None or value > best[0]:
            best = (value, seed, image)
    if best is not None:
        tmp.write_bytes(best[2])
        log.info("%s: kept seed %d after %d attempts (characters %d)", label, best[1], attempts, best[0])


def _draw_style_reference(
    folder: Path,
    title: str | None,
    texts: list[str],
    people: list[Character],
    style: Style,
    *,
    tier: Tier,
) -> tuple[int, int, list[dict]]:
    """Draw `style.png` if it is missing. Returns drawn, normalized, failures.

    The moment is an iconic scene from the whole storyboard. Casting and
    the prompt are the same steps a scene uses. With any quality
    threshold, seeds run from 1 and the picture is checked like a scene.
    There is no style picture yet, so the style criterion is not scored.
    A file already there is kept.
    """
    path = folder / _STYLE_REFERENCE
    if path.is_file():
        normalized = 1 if _normalize_once(path, style) else 0
        return 0, normalized, []
    log.info("Drawing the style reference")
    tmp = path.with_name(f".{path.stem}.partial.png")
    try:
        moment = scenes.choose_iconic_scene(title, texts, people, tier=tier)
        story = characters.story_text(texts)
        surrounding = characters.cut(story)
        picked = scenes.pick_cast(title, surrounding, moment, people, tier=tier)
        chosen = _chosen(people, picked.character_ids)
        prompt = scenes.write_prompt(
            title,
            story,
            moment,
            chosen,
            tier=tier,
            monochrome=style.monochrome,
        ).prompt.strip()
        report: consistency.Report | None = None
        if style.thresholds.enabled():
            prompt, report = _draw_until_accepted(
                title,
                story,
                moment,
                chosen,
                tmp,
                style,
                prompt,
                tier=tier,
                style_reference=None,
                label="style",
            )
        else:
            prompt, _blocked = drawing.draw_scene(
                title,
                story,
                moment,
                chosen,
                tmp,
                style,
                prompt,
                tier=tier,
                seed=1,
            )
            _normalize_once(tmp, style)
        _stamp(tmp, **_prompt_marks(prompt, report))
        tmp.replace(path)
    except Exception as exc:
        log.error("style reference failed: %s", _error(exc))
        return 0, 0, [{"error": _error(exc)}]
    finally:
        tmp.unlink(missing_ok=True)
    return 1, 1 if style.normalize is not None else 0, []


def _draw_scenes(
    folder: Path,
    title: str | None,
    texts: list[str],
    people: list[Character],
    style: Style,
    *,
    tier: Tier,
    seed: int | None,
    style_reference: Path,
) -> tuple[int, int, list[dict]]:
    """One PNG per scene. Returns how many were drawn, how many normalized, and failures.

    Prompts are written in scene order. Each new prompt sees the prompts
    already stored on earlier pictures.
    """
    drawn = 0
    normalized = 0
    failed: list[dict] = []
    story = characters.story_text(texts)
    earlier: list[tuple[int, str]] = []
    for index, scene in enumerate(texts):
        number = index + 1
        path = folder / f"{number}.png"
        try:
            if path.is_file() and _scene_current(path, scene):
                if _normalize_once(path, style):
                    normalized += 1
                _remember_scene(path, scene)
                saved = _tags(path).get(_PROMPT, "").strip()
                if saved:
                    earlier.append((number, saved))
                continue
            log.info("Drawing scene %d of %d", number, len(texts))
            prompt = _draw_one_scene(
                folder,
                title,
                texts,
                index,
                people,
                style,
                story=story,
                earlier=earlier,
                tier=tier,
                seed=seed,
                style_reference=style_reference,
            )
            if prompt.strip():
                earlier.append((number, prompt.strip()))
            drawn += 1
            if style.normalize is not None:
                normalized += 1
        except Exception as exc:
            log.error("scene %d failed: %s", number, _error(exc))
            failed.append({"index": number, "error": _error(exc)})
    return drawn, normalized, failed


def _draw_one_scene(
    folder: Path,
    title: str | None,
    texts: list[str],
    index: int,
    people: list[Character],
    style: Style,
    story: str,
    earlier: Sequence[tuple[int, str]],
    *,
    tier: Tier,
    seed: int | None,
    style_reference: Path,
) -> str:
    """Prompt this scene and draw it. Publish the PNG only once it is chosen.

    Returns the prompt that was drawn. Attempts land in a partial file.
    The numbered PNG is replaced only after normalize and, when required,
    a quality check. A passing picture is kept at once. A miss stops the
    later checks. After the style's `attempts` pictures, none of which
    passed, every criterion is scored on every attempt and the one with
    the best geometric mean is kept. A stopped run therefore leaves the
    previous picture in place and starts its seeds again; seeds already
    drawn come back from the image cache.
    """
    scene = texts[index]
    number = index + 1
    path = folder / f"{number}.png"
    surrounding = scenes.context(texts, index)
    picked = scenes.pick_cast(title, surrounding, scene, people, tier=tier)
    chosen = _chosen(people, picked.character_ids)
    prompt = scenes.write_prompt(
        title,
        story,
        scene,
        chosen,
        tier=tier,
        scene_number=number,
        earlier=earlier,
        monochrome=style.monochrome,
    ).prompt.strip()

    # Any threshold starts at seed 1 and tries at most style.attempts seeds.
    # Without a threshold, the caller's seed is used once.
    tmp = path.with_name(f".{path.stem}.partial.png")
    report: consistency.Report | None = None
    try:
        if style.thresholds.enabled():
            prompt, report = _draw_until_accepted(
                title,
                story,
                scene,
                chosen,
                tmp,
                style,
                prompt,
                tier=tier,
                style_reference=style_reference,
                label=f"scene {number}",
                scene_number=number,
                earlier=earlier,
            )
        else:
            prompt, _blocked = drawing.draw_scene(
                title,
                story,
                scene,
                chosen,
                tmp,
                style,
                prompt,
                tier=tier,
                style_reference=style_reference,
                seed=seed,
                scene_number=number,
                earlier=earlier,
            )
            _normalize_once(tmp, style)
        _remember_scene(tmp, scene, prompt, report)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return prompt


def _draw_until_accepted(
    title: str | None,
    story: str,
    scene: str,
    chosen: list[Character],
    tmp: Path,
    style: Style,
    prompt: str,
    *,
    tier: Tier,
    style_reference: Path | None,
    label: str,
    scene_number: int | None = None,
    earlier: Sequence[tuple[int, str]] = (),
) -> tuple[str, consistency.Report | None]:
    """Draw seeds 1..style.attempts. Keep the first pass, else the best mean.

    Returns the prompt of the picture left in `tmp`, and its scores.

    A score under its mark stops the later checks. The picture left in
    `tmp` is the one to publish. If none pass, every skipped criterion is
    scored on every attempt, and the highest geometric mean is kept. An
    equal mean keeps the earlier seed. The returned prompt is the one
    that picture was drawn from, including a rewrite after a refusal.
    """
    drawn: list[tuple[int, bytes, str, consistency.Report]] = []
    for attempt in range(1, style.attempts + 1):
        prompt, _blocked = drawing.draw_scene(
            title,
            story,
            scene,
            chosen,
            tmp,
            style,
            prompt,
            tier=tier,
            style_reference=style_reference,
            seed=attempt,
            scene_number=scene_number,
            earlier=earlier,
        )
        _normalize_once(tmp, style)
        ok, report = _picture_ok(style, style_reference, tmp, chosen, prompt, tier, label=label, seed=attempt)
        if ok:
            return prompt, report
        drawn.append((attempt, tmp.read_bytes(), prompt, report))
    if any(item[3].pending(style.thresholds) for item in drawn):
        log.info("%s: no picture passed; scoring every attempt", label)
    best: tuple[float, int, bytes, str, consistency.Report] | None = None
    for seed, image, drawn_prompt, report in drawn:
        if report.pending(style.thresholds):
            tmp.write_bytes(image)
            _ok, report = _picture_ok(
                style,
                style_reference,
                tmp,
                chosen,
                drawn_prompt,
                tier,
                label=label,
                seed=seed,
                stop_on_fail=False,
                prior=report,
            )
        mean = report.geometric_mean()
        if mean is not None and (best is None or mean > best[0]):
            best = (mean, seed, image, drawn_prompt, report)
    if best is not None:
        tmp.write_bytes(best[2])
        log.info("%s: kept seed %d after %d attempts (geometric mean %.1f)", label, best[1], style.attempts, best[0])
        return best[3], best[4]
    if drawn:
        tmp.write_bytes(drawn[-1][1])
        return drawn[-1][2], drawn[-1][3]
    return prompt, None


def _chosen(people: Sequence[Character], ids: Sequence[str]) -> list[Character]:
    by_id = {person.id: person for person in people}
    return [by_id[ident] for ident in ids if ident in by_id]


def _picture_ok(
    style: Style,
    reference: Path | None,
    path: Path,
    chosen: Sequence[Character],
    scene: str,
    tier: Tier,
    *,
    label: str,
    seed: int | None,
    stop_on_fail: bool = True,
    prior: consistency.Report | None = None,
) -> tuple[bool, consistency.Report]:
    """Whether every set threshold is met, and the scores behind that.

    A miss stops the later checks. `prior` fills in only the checks that
    were skipped. A criterion the model cannot score does not fail the
    picture, so a failing model does not redraw forever. The geometric
    mean is logged once every set criterion has been scored.
    """

    def _log_score(name: str, score: consistency.Score) -> None:
        need = getattr(style.thresholds, name)
        shown = "-" if score.value is None else str(score.value)
        line = f"{label} seed {seed}: {name} {shown}/{need}"
        if score.reason:
            line += f" - {score.reason}"
        log.debug(line)

    report = consistency.check(
        path,
        thresholds=style.thresholds,
        tier=tier,
        style_reference=reference,
        characters=chosen,
        scene=scene,
        on_score=_log_score,
        stop_on_fail=stop_on_fail,
        prior=prior,
    )
    ok = report.accepted(style.thresholds)
    text = consistency.summary(report, style.thresholds, ok=ok, reasons=False)
    if not report.pending(style.thresholds):
        mean = report.geometric_mean()
        if mean is not None:
            head, sep, tail = text.partition("\n")
            text = f"{head} (geometric mean {mean:.1f}){sep}{tail}"
    log.info("%s seed %s: %s", label, seed, text)
    return ok, report


def _scene_current(path: Path, scene: str) -> bool:
    """True when this PNG was drawn for `scene`, or carries no hash.

    A picture with no hash was put there by hand. It is kept, then stamped
    with the current text. A hash of different text is redrawn.
    """
    stored = _tags(path).get(_SCENE)
    return stored is None or stored == _fingerprint(scene)


def _remember_scene(
    path: Path,
    scene: str,
    prompt: str | None = None,
    report: consistency.Report | None = None,
) -> None:
    """Record the scene text this picture belongs to, the prompt, and its scores.

    The normalize mark already on the file is kept. Omitting `prompt` or
    `report` leaves what is already stored on the picture alone.
    """
    token = _fingerprint(scene)
    tags = _tags(path)
    extra: dict[str, str] = {}
    if tags.get(_SCENE) != token:
        extra[_SCENE] = token
    for key, value in _prompt_marks(prompt, report).items():
        if tags.get(key) != value:
            extra[key] = value
    if extra:
        _stamp(path, **extra)


def _prompt_marks(prompt: str | None, report: consistency.Report | None) -> dict[str, str]:
    """The prompt and score marks for a picture, leaving out what is empty."""
    marks: dict[str, str] = {}
    cleaned = prompt.strip() if prompt else ""
    if cleaned:
        marks[_PROMPT] = cleaned
    scores = consistency.report_scores(report) if report is not None else {}
    if scores:
        marks[_SCORES] = json.dumps(scores, ensure_ascii=False)
    return marks


def marks(path: str | Path) -> tuple[str, dict[str, consistency.Score]]:
    """The prompt a picture was drawn from, and the quality scores stored with it.

    The scores are empty for a picture drawn with no quality threshold.
    """
    tags = _tags(Path(path))
    try:
        raw = json.loads(tags.get(_SCORES) or "{}")
    except json.JSONDecodeError:
        raw = {}
    scores: dict[str, consistency.Score] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            score = consistency.from_dict(value)
            if score is not None:
                scores[name] = score
    return tags.get(_PROMPT, "").strip(), scores


def _normalize_once(path: Path, style: Style) -> bool:
    """Nudge `path` onto the style, once. Returns whether this call changed it."""
    if style.normalize is None or not path.is_file():
        return False
    tags = _tags(path)
    if tags.get(_NORMALIZED) == "1":
        return False
    normalize.apply(path, style.normalize, monochrome=style.monochrome)
    tags[_NORMALIZED] = "1"
    _write_tags(path, tags)
    return True


def _fingerprint(scene: str) -> str:
    return hashlib.sha256(" ".join(scene.split()).encode("utf-8")).hexdigest()


def _tags(path: Path) -> dict[str, str]:
    try:
        with Image.open(path) as img:
            info = img.info
            return {
                key: info[key]
                for key in (_SCENE, _NORMALIZED, _PROMPT, _SCORES)
                if isinstance(info.get(key), str)
            }
    except OSError:
        return {}


def _stamp(path: Path, **extra: str) -> None:
    tags = _tags(path)
    tags.update(extra)
    _write_tags(path, tags)


def _write_tags(path: Path, tags: dict[str, str]) -> None:
    with Image.open(path) as img:
        img.load()
        info = PngInfo()
        for key in (_SCENE, _NORMALIZED):
            value = tags.get(key)
            if value:
                info.add_text(key, value)
        for key in (_PROMPT, _SCORES):
            value = tags.get(key)
            if value:
                info.add_itxt(key, value)
        data = io.BytesIO()
        img.save(data, format="PNG", pnginfo=info)
    path.write_bytes(data.getvalue())


def _clear_partials(folder: Path) -> None:
    for directory in (folder, folder / "characters"):
        if not directory.is_dir():
            continue
        for partial in directory.glob(".*.partial.png"):
            partial.unlink(missing_ok=True)


def _exists(path: Path | None) -> bool:
    return path is not None and path.is_file()


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
