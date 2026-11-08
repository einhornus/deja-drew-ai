"""One helper for every typed LLM call: Pydantic AI over OpenRouter.

`ask()` sends a system prompt, a user prompt, and optional images, and
returns an instance of the Pydantic model you asked for. The model walks
the chain from `Settings.llm_models(tier)`: if one model fails (content
filter, provider error, unexpected shape), the next is tried. `ask()`
returns None only when every model failed, and the caller falls back.

Identical calls (same model, prompts, images, and output schema) are
served from a local JSON cache under `<cache_dir>/llm/`.

Prompting: the model reasons in the reply, then we parse. Every output
type declares an `analysis` field first, with a description that says
what to reason about. Provider thinking tokens are not relied on.

Set the `dejadrew.llm` logger to DEBUG to see every prompt and answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from . import config
from .config import Settings, Tier

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _cache_key(
    model: str,
    system: str,
    user: str,
    output_type: type[BaseModel],
    images: tuple[Path, ...],
) -> str:
    """Same model + same prompts + same images + same expected shape -> same entry."""
    request: dict[str, Any] = {
        "model": model,
        "system": system,
        "user": user,
        "schema": output_type.model_json_schema(),
    }
    if images:
        request["images"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_load(path: Path, output_type: type[T]) -> T | None:
    if not path.is_file():
        return None
    try:
        return output_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _cache_save(path: Path, result: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _user_prompt(user: str, images: tuple[Path, ...]) -> Any:
    """What `Agent.run()` receives: the text alone, or the text then each image."""
    if not images:
        return user
    from pydantic_ai import BinaryContent

    parts: list[Any] = [user]
    for path in images:
        media_type = _MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise ValueError(f"Unsupported image type for the LLM: {path}")
        parts.append(BinaryContent(data=path.read_bytes(), media_type=media_type))
    return parts


def _agent(model: str, output_type: type[BaseModel], system: str, settings: Settings) -> Any:
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(
        api_key=settings.api_key,
        app_url=config.APP_URL,
        app_title=config.APP_TITLE,
    )
    return Agent(
        OpenRouterModel(model, provider=provider),
        output_type=output_type,
        system_prompt=system,
        model_settings={"max_tokens": settings.max_tokens},
    )


def _run(agent: Any, prompt: Any) -> Any:
    """`agent.run()` to completion, with its HTTP client closed afterwards.

    Inside a running event loop (a notebook), the call runs on a worker
    thread with its own loop, so `draw()` stays a plain function.
    """

    async def go() -> Any:
        async with agent:
            return await agent.run(prompt)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["result"] = asyncio.run(go())
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _log_call(
    kind: str,
    model: str,
    output_type: type[BaseModel],
    system: str,
    user: str,
    images: tuple[Path, ...],
    *,
    output: BaseModel | None = None,
    error: BaseException | None = None,
) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return
    lines = [
        "=" * 72,
        f"LLM {kind}  model={model}  -> {output_type.__name__}",
        "-- system --",
        system,
        "-- user --",
        user,
    ]
    if images:
        lines.append("-- images --")
        lines.extend(str(p) for p in images)
    if error is not None:
        lines += ["-- error --", f"{type(error).__name__}: {error}"]
    elif output is not None:
        lines += ["-- answer --", output.model_dump_json(indent=2)]
    lines.append("=" * 72)
    log.debug("\n".join(lines))


def ask(
    output_type: type[T],
    system: str,
    user: str,
    tier: Tier = "high",
    images: Sequence[str | Path] = (),
) -> T | None:
    """Ask the LLM for a `output_type` object. None when every model failed.

    `images` are JPEG/PNG/WebP files sent after the user text, in order.
    Every model in the chain must accept image input.
    """
    settings = config.current()
    if not settings.api_key:
        return None
    image_paths = tuple(Path(p) for p in images)
    last_error: BaseException | None = None
    names = settings.llm_models(tier)
    for name in names:
        cache_path = settings.llm_cache_dir / f"{_cache_key(name, system, user, output_type, image_paths)}.json"
        if settings.cache:
            cached = _cache_load(cache_path, output_type)
            if cached is not None:
                _log_call("CACHE", name, output_type, system, user, image_paths, output=cached)
                return cached
        try:
            agent = _agent(name, output_type, system, settings)
            output = _run(agent, _user_prompt(user, image_paths)).output
        except Exception as exc:
            last_error = exc
            _log_call("CALL", name, output_type, system, user, image_paths, error=exc)
            log.info("LLM %s failed: %s: %s", name, type(exc).__name__, exc)
            continue
        _log_call("CALL", name, output_type, system, user, image_paths, output=output)
        if settings.cache:
            _cache_save(cache_path, output)
        return output
    if last_error is not None:
        log.warning("LLM ask: every model failed (%s); using the fallback.", ", ".join(names))
    return None
