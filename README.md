# DejaDrewAI

Consistent storyboards with AI. You give it a storyboard (a list of scene descriptions) and a style file. It gives back one picture per scene, and the same characters look the same in every picture.

It's BYOK: one [OpenRouter](https://openrouter.ai) key pays for both the LLM calls (made through [Pydantic AI](https://ai.pydantic.dev)) and the image model.

```python
import dejadrew

result = dejadrew.draw(
    [
        "A young red fox digs in the snow and uncovers a small brass key.",
        "The fox drops the key at the feet of an old grey owl.",
        "The fox turns the key in a tiny door at the roots of an oak.",
    ],
    style="black_and_white",        # a bundled style, a path to a JSON file, or a dict
    output="out/fox",
    api_key="sk-or-...",            # or set OPENROUTER_API_KEY
)
```

```
out/fox/
  characters/fox.png      one reference portrait per recurring character
  characters/owl.png
  style.png               one iconic scene, the style reference for the whole storyboard
  1.png  2.png  3.png     one picture per scene
```

## Install

Python 3.11+.

In another project, install it like any other package:

```bash
pip install dejadrew                                          # from PyPI
pip install git+https://github.com/einhornus/deja-drew-ai     # straight from GitHub
pip install G:/path/to/deja-drew-ai                           # from a local checkout
```

To work on DejaDrewAI itself, install it in editable mode, so your edits take effect without reinstalling:

```bash
git clone https://github.com/einhornus/deja-drew-ai
cd deja-drew-ai
pip install -e .
python examples/quickstart.py
```

## Settings: arguments or `.env`

Each setting can be passed to `dejadrew.draw`. A setting you leave out is read from the environment, then from the nearest `.env` file in the current folder or above it, then the built-in default. So you can keep everything in `.env` (copy [`.env.example`](.env.example)), pass everything in code, or mix the two:

```python
dejadrew.draw(
    scenes,
    style="black_and_white",
    output="out/fox",
    api_key="sk-or-...",
    cache_dir="D:/caches/dejadrew",
    tier="medium",
    tier_models={
        "medium": ["anthropic/claude-sonnet-4.5", "openai/gpt-5.6-sol"],
        "low": ["google/gemini-3-flash-preview"],
    },
)
```

`tier="medium"` tries the medium list, then the low list. `dejadrew.TIER_MODELS` holds the built-in lists.

## How it stays consistent

1. **Cast.** An LLM reads the whole storyboard and lists the characters who recur, then writes a short, drawable appearance for each one: the big, stable facts (age, build, hair, usual clothes and colours) and nothing that changes from scene to scene.
2. **Portraits.** Each character gets a plain reference sheet, `characters/<id>.png`.
3. **Style reference.** The LLM picks one iconic moment that has as many of the characters as possible, and that moment is drawn as `style.png`.
4. **Scenes.** For each scene the LLM decides who is visible and writes a scene-only prompt. That prompt describes no looks and no style, and it sees every earlier prompt so that neighbouring pictures don't repeat each other. The image model gets the style reference, the portraits of the visible characters, the style paragraph, and the scene prompt, in that order.
5. **Quality loop (optional).** When the style sets thresholds, a vision LLM scores each picture for style match, character match, technical defects, and scene fidelity. A picture that misses a mark is redrawn at the next seed. If no attempt passes, the one with the best geometric mean of its scores is kept.
6. **Normalize (optional).** Each picture is nudged onto the same paper, stroke width, line darkness, and wash tone, so a set of pencil sketches reads as one hand.

The prompt each picture was drawn from, and its scores, are stored inside the PNG:

```python
prompt, scores = dejadrew.marks("out/fox/2.png")
```

## Reruns are cheap

- A picture already on disk is kept. Delete a PNG to redraw it.
- A scene is redrawn when its text changes, because a hash of that text is stored in the PNG.
- LLM answers and seeded images are cached under `.cache/dejadrew/`, so an interrupted run picks up where it stopped without paying twice.

## CLI

```bash
dejadrew draw examples/fox.json --style black_and_white --out out/fox
dejadrew draw scenes.json --style my_style.json --out out/ --tier low -v
dejadrew measure reference.png         # stroke/tone numbers for a style's "normalize" block
dejadrew styles                        # list bundled styles
```

The storyboard file is a JSON array of strings, or `{"title": "...", "scenes": [...]}`. The CLI reads `.env` from the current folder.

## Python API

```python
dejadrew.draw(
    storyboard,                 # list[str], one string per scene
    style,                      # "black_and_white" | "path/to/style.json" | dict
    output,                     # folder
    title=None,                 # optional context for the LLM
    api_key=None,               # falls back to OPENROUTER_API_KEY
    tier="high",                # "high" | "medium" | "low": which LLM list to start from
    tier_models=None,           # {"high": [...], "medium": [...], "low": [...]}, replaces those lists
    models=None,                # your own OpenRouter LLM ids, tried in order, ignoring tiers
    seed=None,                  # image seed when the style sets no thresholds
    cache_dir=None,             # default .cache/dejadrew
    cache=None,                 # False to bypass both caches
    max_tokens=None,            # output budget per LLM call, default 16384
) -> dict
```

`draw` returns a JSON-able summary: the cast with each appearance, what was drawn and what was reused, and any failures. Progress goes to the `dejadrew` logger. Set `dejadrew.llm` to `DEBUG` to see every prompt and answer.

## Style files

```json
{
  "width": 1280,
  "height": 720,
  "model": "google/gemini-3-pro-image",
  "prompt": "Art style: a delicate hand-drawn graphite pencil sketch ...",
  "monochrome": true,
  "attempts": 5,
  "thresholds": { "style": 80, "characters": 80, "scene": 80, "technical": 70 },
  "normalize": { "enabled": true, "width_strength": 0.7 }
}
```

| key | |
|---|---|
| `width`, `height` | Required. Output size. The nearest supported ratio is requested, then cropped. |
| `model` | Required. Any OpenRouter image model, e.g. `google/gemini-3-pro-image`, `openai/gpt-image-2`. |
| `prompt` | Required. The artistic-style paragraph sent with every picture. |
| `quality` | `low`, `medium`, `high`, `auto`. Required for OpenAI image models. |
| `monochrome` | When `true`, the descriptions and prompts the LLM writes name no colours. |
| `thresholds` | 0–100 pass marks for `style`, `characters`, `scene`, `technical`. Leave it out to skip the quality loop (much cheaper). |
| `attempts` | Seeds per picture in the quality loop (default 5). Portraits get 3× that. |
| `normalize` | Post-processing toward a fixed drawing look. Measure a reference with `dejadrew measure`. `false` disables it. |

The bundled style is [`dejadrew/styles/black_and_white.json`](dejadrew/styles/black_and_white.json).

## Cost

Without `thresholds`, a run makes one LLM call to list the cast, one per character, 2 per scene, and 3 for the style reference, plus one image per character, one per scene, and one for the style reference. With `thresholds`, each picture can take up to `attempts` images and 4 vision scores per image. Start with `--tier low` and no thresholds while you iterate on the style.

## Environment

| variable | |
|---|---|
| `OPENROUTER_API_KEY` | Your key. |
| `DEJADREW_LLM_MODELS` | Comma-separated OpenRouter LLM ids, replacing the tier lists. |
| `DEJADREW_MODELS_HIGH`, `DEJADREW_MODELS_MEDIUM`, `DEJADREW_MODELS_LOW` | Comma-separated OpenRouter LLM ids, replacing one tier list. |
| `DEJADREW_CACHE_DIR` | Cache folder (default `.cache/dejadrew`). |
| `DEJADREW_CACHE` | `0` turns caching off. |
| `DEJADREW_MAX_TOKENS` | Output budget per LLM call (default 16384). |

These can live in a `.env` file. Arguments passed to `draw` win over the environment, and real environment variables win over `.env`.

## Building and publishing

```powershell
.\build.ps1                  # builds dist/dejadrew-<version>.tar.gz and .whl, and checks them
.\build.ps1 -Publish -Test   # uploads to TestPyPI, a sandbox, to try the release first
.\build.ps1 -Publish         # uploads to PyPI; then anyone can `pip install dejadrew`
```

One-time setup: create accounts on [pypi.org](https://pypi.org/account/register/) (and [test.pypi.org](https://test.pypi.org/account/register/) if you want the sandbox), then create an API token under Account settings, API tokens. When the upload asks, enter `__token__` as the username and the token as the password.

Before each release, bump `__version__` in [`dejadrew/__init__.py`](dejadrew/__init__.py). PyPI never accepts the same version twice.

You can also share the wheel without PyPI: `pip install dist/dejadrew-0.1.0-py3-none-any.whl`.

## License

MIT
