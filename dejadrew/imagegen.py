"""Image generation through OpenRouter. Thin HTTP wrapper, no SDK.

Both image families a style file can name (Google's Nano Banana and
OpenAI's GPT Image) are chat-completion models on OpenRouter that answer
with `message.images`, so one request shape covers them and the same
OpenRouter key that pays for the LLM calls pays for them.

A call with a `seed` is served from a local cache under
`<cache_dir>/image/`, the same way `llm.ask` serves a repeated prompt.
`seed=None` skips the cache and always draws a new image.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from . import config

URL = "https://openrouter.ai/api/v1/chat/completions"

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_CACHE_SUFFIXES = (".png", ".jpg", ".webp", ".gif", ".bin")


class ImageGenError(RuntimeError):
    """`blocked` is True when the model refused this prompt (moderation, no
    image returned): a different prompt may pass. False for auth, credit,
    rate-limit, and server errors, where rewording changes nothing."""

    def __init__(self, message: str, blocked: bool = False):
        super().__init__(message)
        self.blocked = blocked


# 400 (moderation / content policy), 403 (flagged input), 451 (unavailable for legal reasons)
_BLOCK_STATUSES = {400, 403, 451}


def decode(data: bytes) -> Image.Image:
    """Bytes -> upright RGB image. Raises if the bytes are not an image."""
    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.getchannel("A"))
        return flat
    return img.convert("RGB")


def _data_url(path: Path) -> str:
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ImageGenError(f"Unsupported reference image type: {path}")
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _download(url: str, client: httpx.Client) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    r = client.get(url)
    r.raise_for_status()
    return r.content


def _references(reference: Path | str | Sequence[Path | str] | None) -> list[Path]:
    if reference is None:
        return []
    if isinstance(reference, (str, Path)):
        return [Path(reference)]
    return [Path(path) for path in reference]


def is_openai(model: str) -> bool:
    """True when `model` is an OpenRouter OpenAI slug, such as `openai/gpt-image-2`."""
    return model.split("/", 1)[0].lower() == "openai"


def _cache_key(
    model: str,
    prompt: str,
    seed: int,
    reference: Path | str | Sequence[Path | str] | None,
    aspect_ratio: str,
    style: str | None,
    style_last: bool,
    quality: str | None,
) -> str:
    """Same model + prompt + seed + references + style + ratio + quality -> same file."""
    request: dict = {
        "model": model,
        "prompt": prompt,
        "seed": seed,
        "aspect_ratio": aspect_ratio,
    }
    if style and style.strip():
        request["style"] = style
        request["style_last"] = style_last
    if quality and is_openai(model):
        request["quality"] = quality
    refs = _references(reference)
    if refs:
        request["references"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in refs]
    payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return ".bin"


def _cache_load(directory: Path, key: str) -> bytes | None:
    for suffix in _CACHE_SUFFIXES:
        path = directory / f"{key}{suffix}"
        try:
            if path.is_file():
                data = path.read_bytes()
                if data:
                    return data
        except OSError:
            return None
    return None


def _cache_save(directory: Path, key: str, data: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}{_image_suffix(data)}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def generate(
    model: str,
    prompt: str,
    reference: Path | str | Sequence[Path | str] | None = None,
    aspect_ratio: str = "16:9",
    seed: int | None = None,
    timeout: float = 300.0,
    *,
    style: str | None = None,
    style_last: bool = False,
    quality: str | None = None,
) -> bytes:
    """One image from `model`. Returns the image bytes.

    When `reference` is given (one path or several), the message is those
    images, then `style` if given, then `prompt`. Pass `style_last=True` to
    send style after the prompt instead. `aspect_ratio` goes to OpenRouter's
    `image_config`; the caller crops to fit anyway. `quality` (`low`,
    `medium`, `high`, or `auto`) is added to that config only for OpenAI
    models.

    `seed` is sent when it is an int, and that call is cached: the same
    request returns the saved image without calling the API. `seed=None`
    is not cached and always draws a new image.
    """
    settings = config.current()
    cache_key = None
    if seed is not None and settings.cache:
        cache_key = _cache_key(model, prompt, seed, reference, aspect_ratio, style, style_last, quality)
        cached = _cache_load(settings.image_cache_dir, cache_key)
        if cached is not None:
            return cached
    if not settings.api_key:
        raise ImageGenError("No OpenRouter key. Pass api_key= or set OPENROUTER_API_KEY.")
    content: list[dict] = []
    for path in _references(reference):
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    if style and style.strip() and not style_last:
        content.append({"type": "text", "text": style})
    content.append({"type": "text", "text": prompt})
    if style and style.strip() and style_last:
        content.append({"type": "text", "text": style})
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": aspect_ratio},
    }
    if quality and is_openai(model):
        body["image_config"]["quality"] = quality
    if seed is not None:
        body["seed"] = seed
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "HTTP-Referer": config.APP_URL,
        "X-Title": config.APP_TITLE,
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.post(URL, headers=headers, json=body)
        if r.status_code >= 400:
            raise ImageGenError(
                f"{model} failed ({r.status_code}): {r.text[:400]}",
                blocked=r.status_code in _BLOCK_STATUSES,
            )
        data = r.json()
        if data.get("error"):
            error = data["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise ImageGenError(f"{model} failed: {str(error)[:400]}", blocked=code in _BLOCK_STATUSES)
        message = ((data.get("choices") or [{}])[0]).get("message") or {}
        images = message.get("images") or []
        if not images:
            said = (message.get("content") or "").strip()[:400]
            raise ImageGenError(f"{model} returned no image. Model said: {said or '(nothing)'}", blocked=True)
        image = _download(images[0]["image_url"]["url"], client)
    if cache_key is not None and image:
        _cache_save(settings.image_cache_dir, cache_key, image)
    return image
