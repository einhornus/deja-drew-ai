"""Command line.

    dejadrew draw storyboard.json --style black_and_white --out out/
    dejadrew measure reference.png [more.png ...]
    dejadrew styles

`storyboard.json` is a JSON array of strings, one per scene. An object
`{"title": ..., "scenes": [...]}` works too. The nearest `.env` at or
above the current folder is read, so `OPENROUTER_API_KEY` can live there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import normalize
from .storyboard import draw
from .style import bundled


def _read_storyboard(path: Path) -> tuple[list[str], str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    title = None
    if isinstance(data, dict):
        title = data.get("title")
        data = data.get("scenes")
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise SystemExit(f"{path} must be a JSON array of strings, or an object with a 'scenes' array")
    return data, title if isinstance(title, str) else None


def _draw(args: argparse.Namespace) -> int:
    scenes, title = _read_storyboard(Path(args.storyboard))
    result = draw(
        scenes,
        args.style,
        args.out,
        title=args.title or title,
        api_key=args.api_key,
        tier=args.tier,
        seed=args.seed,
        models=args.models,
        cache_dir=args.cache_dir,
        cache=False if args.no_cache else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _measure(args: argparse.Namespace) -> int:
    styles = [normalize.measure(Path(p)) for p in args.images]
    measured = normalize.average(styles) if len(styles) > 1 else styles[0]
    keys = ("ref_width", "stroke_width", "line_values", "wash_values")
    full = asdict(measured)
    print(json.dumps({key: full[key] for key in keys}, indent=2))
    return 0


def _styles(_args: argparse.Namespace) -> int:
    print("\n".join(bundled()))
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    parser = argparse.ArgumentParser(prog="dejadrew", description="Consistent storyboards with AI.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for progress details, -vv for every LLM prompt")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("draw", help="draw one picture per scene of a storyboard")
    p.add_argument("storyboard", help="JSON array of scene strings")
    p.add_argument("--style", required=True, help="style JSON path, or a bundled style name")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--title", help="optional story title, context for the LLM")
    p.add_argument("--api-key", help="OpenRouter key (default: OPENROUTER_API_KEY)")
    p.add_argument("--tier", choices=("high", "medium", "low"), default="high", help="LLM cost tier")
    p.add_argument("--models", help="comma-separated OpenRouter LLM ids, replacing the tier list")
    p.add_argument("--seed", type=int, help="image seed when the style sets no thresholds")
    p.add_argument("--cache-dir", help="answer and image cache (default .cache/dejadrew)")
    p.add_argument("--no-cache", action="store_true", help="ignore and do not write the caches")
    p.set_defaults(run=_draw)

    p = sub.add_parser("measure", help="measure a reference drawing for a style's normalize block")
    p.add_argument("images", nargs="+", help="one or more reference images; several are averaged")
    p.set_defaults(run=_measure)

    p = sub.add_parser("styles", help="list the bundled styles")
    p.set_defaults(run=_styles)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    logging.getLogger("dejadrew").setLevel(logging.DEBUG if args.verbose >= 1 else logging.INFO)
    logging.getLogger("dejadrew.llm").setLevel(logging.DEBUG if args.verbose >= 2 else logging.INFO)
    try:
        return args.run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"dejadrew: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
