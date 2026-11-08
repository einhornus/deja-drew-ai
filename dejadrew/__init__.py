"""DejaDrewAI: consistent storyboards with AI. Bring your own OpenRouter key.

    import dejadrew

    result = dejadrew.draw(
        ["A fox finds a lost key in the snow.", "The fox opens a tiny door in an oak."],
        style="black_and_white",
        output="out/fox",
        api_key="sk-or-...",
    )
"""

from .config import TIER_MODELS, Settings, Tier
from .imagegen import ImageGenError
from .storyboard import draw, marks
from .style import Style, bundled as bundled_styles, load as load_style

__version__ = "0.1.0"

__all__ = [
    "ImageGenError",
    "Settings",
    "Style",
    "TIER_MODELS",
    "Tier",
    "bundled_styles",
    "draw",
    "load_style",
    "marks",
]
