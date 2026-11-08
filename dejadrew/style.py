"""Load a style: canvas size, image model, style prompt, and quality knobs.

A style is a JSON object. Pass `load` a path to a `.json` file, the name
of a bundled style (`black_and_white`, from `dejadrew/styles/`), or the
already-parsed dict.

    width, height   required. The PNG size. The request asks for the
                    nearest ratio OpenRouter accepts, then crops to fit.
    model           required. An OpenRouter image model, such as
                    `google/gemini-3-pro-image` or `openai/gpt-image-2`.
    prompt          required. The artistic-style paragraph sent with
                    every picture.
    quality         `low`, `medium`, `high`, or `auto`. Required for an
                    OpenAI image model and sent only to those.
    monochrome      true tells the appearance and scene-prompt calls the
                    picture is black and white, so those descriptions
                    name no colours. Default false.
    thresholds      optional 0-100 pass marks: `style`, `characters`,
                    `scene`, `technical`. See below.
    attempts        seeds drawn per picture when any threshold is set.
                    Default 5.
    normalize       optional `DrawingStyle` object. See below.

`thresholds`: each set mark scores that criterion on every new scene and
on the style reference. Checks run in the order style, characters,
technical, scene, and a score under its mark skips the rest. The picture
is redrawn with the next seed until every set mark is met, or `attempts`
pictures have been drawn. If none pass, the skipped checks are run on
every attempt and the picture with the best geometric mean of its scores
is kept. The style reference has no picture to compare against yet, so
its style mark is not scored. `null` on one key skips that criterion.
Omitting `thresholds`, or setting it to null, skips the check.

A character portrait is scored against its appearance only when
`thresholds.characters` is set. That bar is stricter than the scene mark
T: `sqrt(T/100)*100`, rounded. The portrait may be redrawn up to three
times `attempts`.

`normalize`: the paper, stroke width, line darkness, and wash tone every
portrait and scene picture is nudged toward. `enabled` inside it says
whether to normalize at all, and defaults to true. `enabled` false,
`normalize` false, or omitting `normalize`, leaves every picture as the
model drew it; the other keys are still checked. Measure a reference
with `dejadrew measure ref.png` (or `normalize.measure`) and paste
`stroke_width`, `line_values`, `wash_values`, and `ref_width` here.
"""

from __future__ import annotations

import json
import math
import typing
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path

from .imagegen import is_openai
from .normalize import DrawingStyle

# GPT Image quality. OpenRouter ignores it for providers without that knob,
# so it is sent only when the style's model is OpenAI.
_QUALITIES = ("auto", "low", "medium", "high")

# Seeds drawn for one picture when any quality threshold is set.
_DEFAULT_ATTEMPTS = 5

# A reference portrait gets this many times as many seeds as a scene.
_PORTRAIT_ATTEMPT_FACTOR = 3

# OpenRouter image_config accepts these ratios. A style's width/height is
# snapped to the nearest one; the PNG is cropped to the exact pixels.
_STANDARD = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
    (21, 9),
)

# Pass marks for the scene quality check. None skips that criterion.
_CRITERIA = ("style", "characters", "scene", "technical")


@dataclass(frozen=True)
class Thresholds:
    style: int | None = None
    characters: int | None = None
    scene: int | None = None
    technical: int | None = None

    def enabled(self) -> bool:
        """True when at least one criterion has a pass mark."""
        return any(getattr(self, name) is not None for name in _CRITERIA)


@dataclass(frozen=True)
class Style:
    name: str
    width: int
    height: int
    model: str
    prompt: str
    aspect_ratio: str
    quality: str | None = None
    attempts: int = _DEFAULT_ATTEMPTS
    thresholds: Thresholds = field(default_factory=Thresholds)
    normalize: DrawingStyle | None = None
    monochrome: bool = False


def portrait_threshold(mark: int) -> int:
    """Character bar for a reference portrait, stricter than the scene mark.

    `sqrt(mark / 100) * 100`, rounded to the nearest integer from 0 to 100.
    Scene pictures keep `mark`.
    """
    raised = ((mark / 100) ** 0.5) * 100
    return max(0, min(100, round(raised)))


def portrait_attempts(attempts: int) -> int:
    """Seeds for one reference portrait: three times a scene's `attempts`."""
    return attempts * _PORTRAIT_ATTEMPT_FACTOR


def aspect_ratio(width: int, height: int) -> str:
    """Nearest OpenRouter ratio to `width`:`height`."""
    target = width / height
    best = min(_STANDARD, key=lambda pair: abs(pair[0] / pair[1] - target))
    g = math.gcd(width, height)
    exact = (width // g, height // g)
    if exact in _STANDARD:
        return f"{exact[0]}:{exact[1]}"
    return f"{best[0]}:{best[1]}"


def bundled() -> list[str]:
    """Names of the styles shipped in `dejadrew/styles/`."""
    folder = resources.files("dejadrew") / "styles"
    return sorted(item.name[:-5] for item in folder.iterdir() if item.name.endswith(".json"))


def load(source: str | Path | dict | Style) -> Style:
    """A style from a JSON path, a bundled style name, or a parsed dict."""
    if isinstance(source, Style):
        return source
    if isinstance(source, dict):
        return parse(source, name="style")
    raw = str(source).strip()
    if not raw:
        raise ValueError("style is empty")
    path = Path(raw).expanduser()
    if path.is_file():
        return parse(_read(path), name=path.stem, where=str(path))
    stem = path.stem if path.suffix.lower() == ".json" else path.name
    if path.parent == Path(".") and stem in bundled():
        text = (resources.files("dejadrew") / "styles" / f"{stem}.json").read_text(encoding="utf-8")
        return parse(_parse_json(text, stem), name=stem, where=f"bundled style {stem}")
    names = ", ".join(bundled()) or "(none)"
    raise FileNotFoundError(f"No style file {path}, and no bundled style of that name ({names})")


def _read(path: Path) -> dict:
    return _parse_json(path.read_text(encoding="utf-8"), str(path))


def _parse_json(text: str, where: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a JSON object")
    return data


def parse(data: dict, *, name: str = "style", where: str | None = None) -> Style:
    """A `Style` from a parsed JSON object. `where` names it in error messages."""
    path = where or name
    try:
        width = int(data["width"])
        height = int(data["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} needs integer width and height") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} width and height must be positive")
    model = (data.get("model") or "").strip()
    if not model:
        raise ValueError(f"{path} needs a model")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path} needs a style prompt")
    return Style(
        name=name,
        width=width,
        height=height,
        model=model,
        prompt=prompt.strip(),
        aspect_ratio=aspect_ratio(width, height),
        quality=_quality(path, data.get("quality"), model),
        attempts=_attempts(path, data.get("attempts")),
        thresholds=_thresholds(path, data.get("thresholds")),
        normalize=_normalize(path, data.get("normalize")),
        monochrome=_monochrome(path, data),
    )


def _monochrome(path: str, data: dict) -> bool:
    """`monochrome` from the style. Omitted means colour is allowed."""
    if "monochrome" not in data:
        return False
    raw = data["monochrome"]
    if not isinstance(raw, bool):
        raise ValueError(f"{path} monochrome must be true or false")
    return raw


def _quality(path: str, raw, model: str) -> str | None:
    """`quality` from the style. Required for an OpenAI model."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if is_openai(model):
            names = ", ".join(_QUALITIES)
            raise ValueError(f"{path} needs quality ({names}) for an OpenAI image model")
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{path} quality must be a string")
    quality = raw.strip().lower()
    if quality not in _QUALITIES:
        names = ", ".join(_QUALITIES)
        raise ValueError(f"{path} quality must be one of: {names}")
    return quality


def _number(path: str, key: str, raw) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{path} normalize {key} must be a number")
    return float(raw)


def _numbers(path: str, key: str, raw, count: int | None = None) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} normalize {key} must be a non-empty list of numbers")
    if count is not None and len(raw) != count:
        raise ValueError(f"{path} normalize {key} must have {count} numbers")
    return tuple(_number(path, key, item) for item in raw)


def _attempts(path: str, raw) -> int:
    """How many seeds to draw for one picture. Omitted uses the default."""
    if raw is None:
        return _DEFAULT_ATTEMPTS
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{path} attempts must be a positive integer")
    return raw


def _normalize(path: str, raw) -> DrawingStyle | None:
    """`normalize` from the style.

    Omitted, false, or `enabled: false` leaves pictures as drawn. The
    other keys are still checked when `enabled` is false.
    """
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} normalize must be an object of style parameters, or false")
    raw = dict(raw)
    enabled = raw.pop("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{path} normalize enabled must be true or false")
    known = {item.name for item in fields(DrawingStyle)}
    unknown = sorted(set(raw) - known)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"{path} normalize has unknown keys: {names}")
    hints = typing.get_type_hints(DrawingStyle)
    kwargs: dict = {}
    for key, value in raw.items():
        kwargs[key] = _coerce(path, key, value, hints[key])
    style = DrawingStyle(**kwargs)
    _check_style(path, style)
    if not enabled:
        return None
    return style


def _coerce(path: str, key: str, raw, annotation):
    """One `DrawingStyle` field from JSON."""
    if annotation is int:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{path} normalize {key} must be a positive integer")
        return raw
    origin = typing.get_origin(annotation)
    if origin is tuple:
        args = typing.get_args(annotation)
        count = None if not args or args[-1] is Ellipsis else len(args)
        return _numbers(path, key, raw, count)
    return _number(path, key, raw)


def _increasing(path: str, name: str, values: tuple[float, ...]) -> None:
    if any(q <= 0.0 or q >= 1.0 for q in values):
        raise ValueError(f"{path} normalize {name} must be between 0 and 1")
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"{path} normalize {name} must increase")


def _unit(path: str, name: str, values: tuple[float, ...]) -> None:
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError(f"{path} normalize {name} must be from 0 to 1")


def _check_style(path: str, style: DrawingStyle) -> None:
    if not 0.0 < style.bg <= 1.0:
        raise ValueError(f"{path} normalize bg must be between 0 and 1")
    if not 0.0 <= style.ink_threshold < style.bg:
        raise ValueError(f"{path} normalize ink_threshold must be from 0 up to bg")
    if style.wash_radius <= 0.0:
        raise ValueError(f"{path} normalize wash_radius must be positive")
    if style.stroke_width <= 0.0:
        raise ValueError(f"{path} normalize stroke_width must be positive")
    if style.max_width_change < 0.0:
        raise ValueError(f"{path} normalize max_width_change must be non-negative")
    for name, value in (
        ("line_threshold", style.line_threshold),
        ("wash_threshold", style.wash_threshold),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{path} normalize {name} must be from 0 up to 1")
    if len(style.line_quantiles) != len(style.line_values):
        raise ValueError(f"{path} normalize line_quantiles and line_values must be the same length")
    if len(style.wash_quantiles) != len(style.wash_values):
        raise ValueError(f"{path} normalize wash_quantiles and wash_values must be the same length")
    _increasing(path, "line_quantiles", style.line_quantiles)
    _increasing(path, "wash_quantiles", style.wash_quantiles)
    _unit(path, "line_values", style.line_values)
    _unit(path, "wash_values", style.wash_values)
    for name, value in (
        ("width_strength", style.width_strength),
        ("line_strength", style.line_strength),
        ("wash_strength", style.wash_strength),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{path} normalize {name} must be from 0 to 1")
    for name, (lo, hi) in (
        ("width_ratio_limits", style.width_ratio_limits),
        ("slope_limits", style.slope_limits),
    ):
        if lo <= 0.0 or lo > hi:
            raise ValueError(f"{path} normalize {name} must be a positive low, high pair")
    lo, hi = style.density_guard
    if lo < 0.0 or hi > 1.0 or lo >= hi:
        raise ValueError(f"{path} normalize density_guard must be a low, high pair inside 0 to 1")
    near, far = style.snap
    if near < 0.0 or far <= near:
        raise ValueError(f"{path} normalize snap must be a near, far pair with near < far")


def _thresholds(path: str, raw) -> Thresholds:
    """`thresholds` from the style. Omitted or null skips every check."""
    if raw is None:
        return Thresholds()
    if not isinstance(raw, dict):
        raise ValueError(f"{path} thresholds must be an object of 0-100 scores, or null")
    unknown = sorted(set(raw) - set(_CRITERIA))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"{path} thresholds has unknown keys: {names}")
    return Thresholds(
        **{name: _bound(path, f"thresholds.{name}", raw[name]) if name in raw else None for name in _CRITERIA}
    )


def _bound(path: str, key: str, raw) -> int | None:
    """One 0-100 threshold. `null` skips that criterion."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{path} {key} must be an integer from 0 to 100, or null")
    if raw < 0 or raw > 100:
        raise ValueError(f"{path} {key} must be from 0 to 100, or null")
    return raw
