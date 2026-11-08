"""Settings for one run: the OpenRouter key, the LLM chain, and the caches.

Bring your own key. Pass `api_key=` to `dejadrew.draw`, or set
`OPENROUTER_API_KEY` in the environment. The same key pays for the LLM
calls (through Pydantic AI) and for the image model (through OpenRouter's
chat-completions endpoint).

Every setting passed to `draw` wins. A missing one is read from the
environment, then from the nearest `.env` file at or above the current
folder, then the built-in default:

    OPENROUTER_API_KEY     the key
    DEJADREW_LLM_MODELS    comma-separated OpenRouter ids, tried in order,
                           instead of the tier lists
    DEJADREW_MODELS_HIGH   comma-separated OpenRouter ids replacing one
    DEJADREW_MODELS_MEDIUM built-in tier list
    DEJADREW_MODELS_LOW
    DEJADREW_CACHE_DIR     where answers and images are cached
                           (default `.cache/dejadrew`)
    DEJADREW_CACHE         `0` turns both caches off
    DEJADREW_MAX_TOKENS    output budget of one LLM call (default 16384)

`draw` resolves these once and installs them for the run with `use`, so
nested calls read them with `current` instead of passing them around.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from dotenv import find_dotenv, load_dotenv

Tier = Literal["high", "medium", "low"]
TIERS: tuple[Tier, ...] = get_args(Tier)

# OpenRouter models for each cost tier, first choice first. When a model
# fails, the next one is tried, then the next cheaper tier.
TIER_MODELS: dict[Tier, tuple[str, ...]] = {
    "high": (
        "anthropic/claude-opus-4.7",
        "openai/gpt-6-astra",
        "google/gemini-4-argon",
        "anthropic/claude-opus-4.8",
        "meta/muse-wonder",
    ),
    "medium": (
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.6-sol",
    ),
    "low": (
        "google/gemini-3-flash-preview",
        "openai/gpt-5.6-luna",
        "google/gemini-3.8-flash",
    ),
}
_NEXT_TIER: dict[Tier, Tier | None] = {"high": "medium", "medium": "low", "low": None}

# Thinking models spend this budget on reasoning before the structured
# answer. The provider default is often too small.
DEFAULT_MAX_TOKENS = 16384
DEFAULT_CACHE_DIR = Path(".cache/dejadrew")

APP_TITLE = "DejaDrewAI"
APP_URL = "https://github.com/einhornus/deja-drew-ai"


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    models: tuple[str, ...] = ()
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache: bool = True
    max_tokens: int = DEFAULT_MAX_TOKENS
    tiers: Mapping[Tier, tuple[str, ...]] = field(default_factory=lambda: dict(TIER_MODELS))

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def image_cache_dir(self) -> Path:
        return self.cache_dir / "image"

    def llm_models(self, tier: Tier) -> list[str]:
        """The chain `ask` walks: `models` when set, else this tier and every cheaper one."""
        if self.models:
            return list(self.models)
        names: list[str] = []
        current: Tier | None = tier
        while current is not None:
            for name in self.tiers.get(current, TIER_MODELS[current]):
                if name not in names:
                    names.append(name)
            current = _NEXT_TIER[current]
        return names


def resolve(
    *,
    api_key: str | None = None,
    models: str | Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    cache: bool | None = None,
    max_tokens: int | None = None,
    tier_models: Mapping[str, str | Sequence[str]] | None = None,
) -> Settings:
    """Settings from the arguments, then the environment and `.env`, then the defaults.

    `tier_models` replaces the model list of each tier it names, e.g.
    `{"low": ["google/gemini-3-flash-preview"]}`. A tier it leaves out comes
    from `DEJADREW_MODELS_<TIER>`, else the built-in list.
    """
    load_env()
    key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip() or None
    if models is None:
        models = os.getenv("DEJADREW_LLM_MODELS", "")
    chain = _chain(models)
    given = dict(tier_models or {})
    unknown = sorted(set(given) - set(TIERS))
    if unknown:
        raise ValueError(f"unknown tier in tier_models: {', '.join(unknown)} (use {', '.join(TIERS)})")
    tiers: dict[Tier, tuple[str, ...]] = {}
    for tier in TIERS:
        names = _chain(given[tier]) if tier in given else _chain(os.getenv(f"DEJADREW_MODELS_{tier.upper()}", ""))
        tiers[tier] = names or TIER_MODELS[tier]
    if cache_dir is None:
        cache_dir = os.getenv("DEJADREW_CACHE_DIR") or DEFAULT_CACHE_DIR
    if cache is None:
        cache = os.getenv("DEJADREW_CACHE", "1").strip().lower() not in ("0", "false", "no")
    if max_tokens is None:
        raw = os.getenv("DEJADREW_MAX_TOKENS", "").strip()
        max_tokens = int(raw) if raw else DEFAULT_MAX_TOKENS
    return Settings(
        api_key=key,
        models=chain,
        cache_dir=Path(cache_dir).expanduser(),
        cache=cache,
        max_tokens=max_tokens,
        tiers=tiers,
    )


_env_loaded = False


def load_env() -> None:
    """Read the nearest `.env` at or above the current folder, once.

    Variables already set in the environment are kept.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def _chain(models: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(models, str):
        models = models.split(",")
    return tuple(_bare(m) for m in models if m and m.strip())


def _bare(model: str) -> str:
    """`openrouter:google/x` and `google/x` both mean the OpenRouter id `google/x`."""
    name = model.strip()
    if name.startswith("openrouter:"):
        name = name[len("openrouter:"):]
    return name


_current: ContextVar[Settings | None] = ContextVar("dejadrew_settings", default=None)


def current() -> Settings:
    """The settings installed by `use`, or ones resolved from the environment."""
    settings = _current.get()
    return settings if settings is not None else resolve()


@contextmanager
def use(settings: Settings) -> Iterator[Settings]:
    token = _current.set(settings)
    try:
        yield settings
    finally:
        _current.reset(token)
