"""Draw a six-scene storyboard in the bundled black-and-white style.

    pip install -e .
    set OPENROUTER_API_KEY=sk-or-...     (or put it in .env)
    python examples/quickstart.py
"""

import json
import logging
from pathlib import Path

import dejadrew

logging.basicConfig(level=logging.INFO, format="%(message)s")

example = "the_north_wind_and_the_sun"
style = "monochrome_sketch.json"
story = json.loads((Path("examples") / example / "story.json").read_text(encoding="utf-8"))

result = dejadrew.draw(
    story["scenes"],
    style= Path("styles") / style,
    output= Path(example) / style,
    title=story["title"],
    tier="high",
)

print(json.dumps(result, indent=2, ensure_ascii=False))

for name in result["wrote"]:
    if name[0].isdigit():
        prompt, scores = dejadrew.marks(Path(result["folder"]) / name)
        print(name, {k: s.value for k, s in scores.items()}, prompt[:80])
