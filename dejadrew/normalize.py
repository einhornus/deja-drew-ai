"""Normalize one picture onto a fixed drawing style.

The picture's darkness d = bg - g is split into two layers:

  * wash — broad shading and the tone under hatching (a morphological
    opening of d, slightly smoothed);
  * line — everything thinner than ~2 * wash_radius (d - wash).

Each layer is nudged toward the style on its own:

  * line width — effective stroke width (ink per unit of stroke length) is
    matched by fractional grayscale dilation/erosion of the line layer. The
    widening fades out where strokes are already dense, so hatching does
    not clog.
  * line darkness — a monotone curve maps the stroke *core* darkness
    quantiles onto the style.
  * wash — a monotone curve maps wash quantiles onto the style.

Statistics are averaged per tile, not per pixel: a large hatched forest
counts like any other region that contains strokes.

A colour picture (`monochrome=False`) is white-balanced to the paper, then
the same steps run on its luminance. Each pixel's colour is kept as a
per-channel tint of the luminance darkness, so the restyled luminance is
the same as the gray result while the hue and saturation stay. Pixels that
stroke widening darkens take their tint from the ink around them. A colour
picture on dark paper is not inverted; it is left unchanged.

All lengths are in pixels at `ref_width` and rescale with the image width.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy.interpolate import PchipInterpolator
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class DrawingStyle:
    # Paper.
    bg: float = 1.0
    ink_threshold: float = 0.06

    # Geometry, in px at ref_width.
    ref_width: int = 1280
    wash_radius: float = 6.0          # strokes narrower than ~2r are "line"
    tile: int = 64                    # statistics are averaged per tile

    # Line layer (darkness: 0 = paper, 1 = black).
    line_threshold: float = 0.06
    stroke_width: float = 2.6         # effective width, px at ref_width
    line_quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    line_values: tuple[float, ...] = (0.25, 0.38, 0.52, 0.64, 0.74)

    # Wash layer.
    wash_threshold: float = 0.03
    wash_quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    wash_values: tuple[float, ...] = (0.05, 0.09, 0.15, 0.24, 0.35)

    # Strengths and clamps. 0 leaves the picture unchanged.
    width_strength: float = 0.8
    line_strength: float = 0.8
    wash_strength: float = 0.6
    width_ratio_limits: tuple[float, float] = (0.6, 1.8)
    max_width_change: float = 1.5     # px at ref_width
    density_guard: tuple[float, float] = (0.35, 0.65)  # stroke coverage
    slope_limits: tuple[float, float] = (0.4, 2.5)
    snap: tuple[float, float] = (0.02, 0.05)


# Same weights as PIL's grayscale conversion (ITU-R 601).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Corners whose channels differ by more than this are scenery, not paper.
_MAX_PAPER_TINT = 0.12


def _load(path: Path, monochrome: bool = True) -> np.ndarray | None:
    """Float image in 0..1 at the file's own size: (h, w) gray, or (h, w, 3) RGB."""
    try:
        img = Image.open(path)
        img.load()
    except (OSError, ValueError):
        return None
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    elif img.mode != "L":
        img = img.convert("RGB")
    if not monochrome:
        return np.asarray(img.convert("RGB")).astype(np.float32) / np.float32(255.0)
    gray = np.asarray(ImageOps.grayscale(img))
    if gray.dtype == np.uint16:
        return gray.astype(np.float32) / np.float32(65535.0)
    return gray.astype(np.float32) / np.float32(255.0)


def _save(img: np.ndarray, path: Path) -> None:
    unit = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if unit.ndim == 2:
        Image.fromarray(unit, mode="L").convert("RGB").save(path, "PNG")
    else:
        Image.fromarray(unit, mode="RGB").save(path, "PNG")


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (rgb @ _LUMA).astype(np.float32)


def estimate_background(g: np.ndarray, patch_frac: float = 0.04) -> tuple[float, float]:
    """Paper value from the corners. One corner that contains drawing is dropped."""
    height, width = g.shape
    patch = max(4, int(min(height, width) * patch_frac))
    patch = min(patch, height, width)
    corners = (
        g[:patch, :patch],
        g[:patch, -patch:],
        g[-patch:, :patch],
        g[-patch:, -patch:],
    )
    medians = np.array([np.median(corner) for corner in corners], dtype=np.float64)
    spreads = np.array(
        [np.percentile(corner, 90) - np.percentile(corner, 10) for corner in corners],
        dtype=np.float64,
    )
    cleanest = np.argsort(spreads)[:3]
    return float(np.median(medians[cleanest])), float(spreads[cleanest].mean())


def _resize_float(img: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    """`size` is (width, height)."""
    unit = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out = Image.fromarray(unit, mode="L").resize(size, resample)
    return np.asarray(out, dtype=np.float32) / np.float32(255.0)


def flatten_paper(g: np.ndarray, bg: float, max_dev: float = 0.08) -> np.ndarray:
    """Uneven paper (a scan, a vignette), clamped so a dark field stays ink."""
    height, width = g.shape
    scale = 512.0 / max(height, width)
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    small = _resize_float(g, (small_w, small_h), Image.Resampling.BOX)
    kernel = min(31, small_w | 1, small_h | 1)
    if kernel >= 3:
        closed = Image.fromarray(np.clip(small * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
        closed = closed.filter(ImageFilter.MaxFilter(kernel))
        closed = closed.filter(ImageFilter.MinFilter(kernel))
        small = np.asarray(closed, dtype=np.float32) / np.float32(255.0)
    sigma = 15.0
    radius = int(3.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    blur = np.exp(-0.5 * (offsets / sigma) ** 2)
    blur /= blur.sum()
    padded = np.pad(small.astype(np.float64), ((0, 0), (radius, radius)), mode="reflect")
    horizontal = sum(padded[:, i : i + small_w] * coeff for i, coeff in enumerate(blur))
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    paper = sum(padded[i : i + small_h] * coeff for i, coeff in enumerate(blur))
    paper = np.clip(paper, bg - max_dev, bg + max_dev)
    return _resize_float(paper.astype(np.float32), (width, height), Image.Resampling.BILINEAR)


def normalize_background(
    g: np.ndarray,
    style: DrawingStyle,
    allow_invert: bool = True,
    flatten: bool = False,
) -> tuple[np.ndarray, dict]:
    """Make the paper `style.bg`. Black stays black."""
    bg, noise = estimate_background(g)
    inverted = False
    if allow_invert and bg < 0.5:
        g = 1.0 - g
        bg = 1.0 - bg
        inverted = True
    paper = flatten_paper(g, bg) if flatten else bg
    out = np.clip(g / np.maximum(paper, 1e-3) * style.bg, 0.0, 1.0).astype(np.float32)
    return out, {
        "detected_bg": round(float(bg), 4),
        "corner_noise": round(float(noise), 4),
        "inverted": inverted,
    }


def normalize_background_color(
    rgb: np.ndarray,
    style: DrawingStyle,
    flatten: bool = False,
) -> tuple[np.ndarray | None, dict]:
    """White-balance the paper to `style.bg`. None when the paper is dark.

    Tinted corners get one gain for all channels, so a coloured scene that
    reaches the corners keeps its hue.
    """
    corners = [estimate_background(rgb[..., c]) for c in range(3)]
    bgs = np.array([bg for bg, _ in corners], dtype=np.float64)
    noise = float(np.mean([n for _, n in corners]))
    luma_bg = float(bgs @ _LUMA)
    tinted = float(bgs.max() - bgs.min()) > _MAX_PAPER_TINT
    report = {
        "detected_bg": [round(float(b), 4) for b in bgs],
        "corner_noise": round(noise, 4),
        "inverted": False,
        "paper_tinted": tinted,
    }
    if luma_bg < 0.5:
        return None, report
    if tinted:
        shared = flatten_paper(_luma(rgb), luma_bg) if flatten else luma_bg
        papers = [shared] * 3
    else:
        papers = [flatten_paper(rgb[..., c], float(bgs[c])) if flatten else float(bgs[c])
                  for c in range(3)]
    out = np.stack(
        [rgb[..., c] / np.maximum(papers[c], 1e-3) * style.bg for c in range(3)], axis=-1
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32), report


def _strictly_increasing(values: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    ordered = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    return ordered + np.arange(len(ordered)) * eps


def _knots(values, top: float) -> np.ndarray:
    """Monotone knots on [0, top], first pinned to 0 and last to `top`."""
    knots = _strictly_increasing(np.concatenate([[0.0], np.asarray(values, np.float64), [top]]))
    knots[-1] = top
    for i in range(len(knots) - 2, -1, -1):
        if knots[i] >= knots[i + 1]:
            knots[i] = knots[i + 1] - 1e-4
    return knots


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _tile_ids(shape: tuple[int, int], tile: int) -> np.ndarray:
    height, width = shape
    nx = (width + tile - 1) // tile
    return (np.arange(height) // tile)[:, None] * nx + (np.arange(width) // tile)[None, :]


def _tile_weights(ids: np.ndarray, min_count: int) -> np.ndarray:
    """Per-sample weights: every tile with enough samples sums to 1."""
    counts = np.bincount(ids)[ids]
    return np.where(counts >= min_count, 1.0 / np.maximum(counts, 1), 0.0)


def _wquantile(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray | None:
    keep = weights > 0
    if not keep.any():
        return None
    v, w = values[keep], weights[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(qs, dtype=np.float64), cdf, v)


def _monotone_lut(src_q, dst_q, strength: float, slope_limits, n: int = 1024):
    """Monotone curve on [0, 1] with 0 -> 0 and 1 -> 1, slope-limited."""
    curve = PchipInterpolator(_knots(src_q, 1.0), _knots(dst_q, 1.0))
    grid = np.linspace(0.0, 1.0, n)
    lut = curve(grid)
    dx = 1.0 / (n - 1)
    lo, hi = slope_limits
    slopes = np.clip(np.diff(lut), lo * dx, hi * dx)
    rebuilt = np.concatenate([[0.0], np.cumsum(slopes)])
    lut = rebuilt / rebuilt[-1]
    return grid, (1.0 - strength) * grid + strength * lut


# --- Layers -----------------------------------------------------------------

def split_layers(g: np.ndarray, style: DrawingStyle, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Darkness split into (wash, line), with wash + line == bg - g."""
    d = np.clip(style.bg - g, 0.0, 1.0).astype(np.float32)
    r = max(1, int(round(style.wash_radius * scale)))
    wash = cv2.morphologyEx(d, cv2.MORPH_OPEN, _disk(r))
    wash = cv2.GaussianBlur(wash, (0, 0), r / 2.0)
    wash = np.minimum(wash, d)
    return wash, d - wash


@dataclass
class LineStats:
    width_px: float          # effective stroke width in image px
    core_q: np.ndarray       # core darkness at style.line_quantiles
    coverage: np.ndarray     # local stroke coverage, 0..1


def line_stats(line: np.ndarray, style: DrawingStyle, scale: float) -> LineStats | None:
    # Half-maximum stroke mask, so width does not depend on darkness.
    peak = cv2.dilate(line, _disk(max(1, int(round(2.0 * scale)))))
    stroke = (line > style.line_threshold) & (line >= 0.5 * peak)
    skel = skeletonize(stroke)
    if int(skel.sum()) < 200:
        return None

    tile = max(8, int(round(style.tile * scale)))
    ids = _tile_ids(line.shape, tile)
    n_tiles = int(ids.max()) + 1
    min_len = max(8, tile // 4)

    support = cv2.dilate(stroke.astype(np.uint8), _disk(1)).astype(bool)
    ink = np.bincount(ids[support], weights=line[support], minlength=n_tiles)
    core_sum = np.bincount(ids[skel], weights=line[skel], minlength=n_tiles)
    length = np.bincount(ids[skel], minlength=n_tiles)
    ok = length >= min_len
    if int(ok.sum()) < 4:
        return None
    # Ink per unit length / core darkness = effective width, one value per tile.
    width = float(np.median(ink[ok] / np.maximum(core_sum[ok], 1e-6)))

    core_q = _wquantile(line[skel], _tile_weights(ids[skel], min_len), style.line_quantiles)
    if core_q is None:
        return None
    coverage = cv2.GaussianBlur(
        stroke.astype(np.float32), (0, 0), max(1.0, 2.0 * style.wash_radius * scale)
    )
    return LineStats(width, core_q, coverage)


def wash_stats(wash: np.ndarray, style: DrawingStyle, scale: float):
    mask = wash > style.wash_threshold
    coverage = float(mask.mean())
    if not mask.any():
        return None, coverage
    tile = max(8, int(round(style.tile * scale)))
    ids = _tile_ids(wash.shape, tile)[mask]
    weights = _tile_weights(ids, max(16, tile * tile // 20))
    if weights.sum() < 3:
        return None, coverage
    return _wquantile(wash[mask], weights, style.wash_quantiles), coverage


# --- Adjustments ------------------------------------------------------------

def change_width(line: np.ndarray, delta_px: float, guard: np.ndarray) -> np.ndarray:
    """Grow (delta > 0) or shrink strokes by `delta_px` total width, fractionally."""
    if abs(delta_px) < 0.05:
        return line
    height, width = line.shape
    ss = int(np.clip(4096 // max(height, width), 1, 4))
    r = abs(delta_px) / 2.0 * ss           # per side, supersampled px
    r0 = int(np.floor(r))
    t = r - r0
    op = cv2.dilate if delta_px > 0 else cv2.erode
    big = cv2.resize(line, (width * ss, height * ss), interpolation=cv2.INTER_LINEAR)

    def at(radius: int) -> np.ndarray:
        out = op(big, _disk(radius)) if radius > 0 else big
        return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)

    moved = (1.0 - t) * at(r0) + t * at(r0 + 1)
    return line + (moved - line) * guard


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def snap_to_white(img: np.ndarray, style: DrawingStyle) -> np.ndarray:
    """Near-paper pixels become paper. A colour pixel snaps on its darkest channel."""
    near, far = style.snap
    key = img if img.ndim == 2 else img.min(axis=-1)
    weight = _smoothstep(style.bg - far, style.bg - near, key)
    if img.ndim == 3:
        weight = weight[..., None]
    return img + (style.bg - img) * weight


def recolor(
    rgb: np.ndarray,
    dark_before: np.ndarray,
    dark_after: np.ndarray,
    style: DrawingStyle,
    scale: float,
) -> np.ndarray:
    """Give the restyled luminance darkness back its colour.

    Each channel's darkness is a tint times the luminance darkness. Where
    the source had ink the tint is the pixel's own; on near-paper pixels it
    is the ink-weighted tint around them, so widened strokes keep their colour.
    """
    dark_c = np.clip(style.bg - rgb, 0.0, 1.0).astype(np.float32)
    own = dark_c / np.maximum(dark_before, 1e-3)[..., None]
    sigma = max(1.0, style.wash_radius * scale)
    near_d = cv2.GaussianBlur(dark_before, (0, 0), sigma)
    near_c = cv2.GaussianBlur(dark_c, (0, 0), sigma)
    around = np.where(
        (near_d > 1e-3)[..., None], near_c / np.maximum(near_d, 1e-3)[..., None], 1.0
    )
    trust = _smoothstep(0.005, 0.03, dark_before)[..., None]
    tint = trust * own + (1.0 - trust) * around
    return np.clip(style.bg - tint * dark_after[..., None], 0.0, 1.0).astype(np.float32)


def apply_array(
    img: np.ndarray,
    style: DrawingStyle,
    *,
    monochrome: bool = True,
    allow_invert: bool = True,
    flatten: bool = False,
) -> tuple[np.ndarray, dict]:
    """Restyle a (h, w) gray image, or with `monochrome=False` a (h, w, 3) RGB one."""
    if monochrome:
        if img.ndim == 3:
            img = _luma(img)
        g, report = normalize_background(img, style, allow_invert, flatten)
        dark, more = _restyle(g, style)
        report.update(more)
        return np.clip(snap_to_white(style.bg - dark, style), 0.0, 1.0).astype(np.float32), report

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    rgb, report = normalize_background_color(img, style, flatten)
    if rgb is None:
        report["skipped"] = "dark paper"
        return img.astype(np.float32), report
    g = _luma(rgb)
    dark, more = _restyle(g, style)
    report.update(more)
    scale = g.shape[1] / style.ref_width
    out = recolor(rgb, np.clip(style.bg - g, 0.0, 1.0), dark, style, scale)
    return np.clip(snap_to_white(out, style), 0.0, 1.0).astype(np.float32), report


def _restyle(g: np.ndarray, style: DrawingStyle) -> tuple[np.ndarray, dict]:
    """Restyled darkness of a gray image whose paper is already `style.bg`."""
    report: dict = {}
    scale = g.shape[1] / style.ref_width
    wash, line = split_layers(g, style, scale)

    stats = line_stats(line, style, scale)
    if stats is not None:
        src_w = stats.width_px / scale
        lo, hi = style.width_ratio_limits
        goal = src_w * float(np.clip((style.stroke_width / src_w) ** style.width_strength, lo, hi))
        goal = float(np.clip(goal, src_w - style.max_width_change, src_w + style.max_width_change))
        if goal > src_w:
            guard = 1.0 - _smoothstep(*style.density_guard, stats.coverage)
        else:
            guard = np.ones_like(line)
        line = change_width(line, (goal - src_w) * scale, guard.astype(np.float32))

        after = line_stats(line, style, scale) or stats
        grid, lut = _monotone_lut(after.core_q, style.line_values,
                                  style.line_strength, style.slope_limits)
        line = np.interp(line, grid, lut).astype(np.float32)
        report.update(
            src_stroke_width=round(src_w, 3),
            goal_stroke_width=round(goal, 3),
            src_line_core=[round(float(q), 3) for q in stats.core_q],
        )
    else:
        report["src_stroke_width"] = None

    wq, wash_cov = wash_stats(wash, style, scale)
    report["wash_coverage"] = round(wash_cov, 4)
    report["src_wash"] = None if wq is None else [round(float(q), 3) for q in wq]
    if wq is not None:
        grid, lut = _monotone_lut(wq, style.wash_values, style.wash_strength, style.slope_limits)
        wash = np.interp(wash, grid, lut).astype(np.float32)

    return np.clip(wash + line, 0.0, style.bg).astype(np.float32), report


def apply(
    path: Path,
    style: DrawingStyle,
    *,
    monochrome: bool,
    allow_invert: bool = True,
    flatten: bool = False,
) -> dict:
    """Restyle the picture at `path` in place. `monochrome` comes from the style file."""
    img = _load(path, monochrome)
    if img is None:
        raise OSError(f"Could not read {path}")
    out, report = apply_array(
        img, style, monochrome=monochrome, allow_invert=allow_invert, flatten=flatten
    )
    _save(out, path)
    return report


# --- Style measurement ------------------------------------------------------

def measure(path: Path, base: DrawingStyle | None = None) -> DrawingStyle:
    """Read stroke width, line core darkness and wash tone from one reference."""
    base = base or DrawingStyle()
    gray = _load(path)
    if gray is None:
        raise OSError(f"Could not read {path}")
    gray, _ = normalize_background(gray, base)
    scale = gray.shape[1] / base.ref_width
    wash, line = split_layers(gray, base, scale)
    stats = line_stats(line, base, scale)
    if stats is None:
        raise ValueError(f"{path.name} has too few strokes to measure a style")
    wq, _ = wash_stats(wash, base, scale)
    return replace(
        base,
        stroke_width=round(stats.width_px / scale, 3),
        line_values=tuple(round(float(v), 3) for v in stats.core_q),
        wash_values=base.wash_values if wq is None else tuple(round(float(v), 3) for v in wq),
    )


def average(styles: Sequence[DrawingStyle]) -> DrawingStyle:
    """Mean of the measured fields. Knobs come from the first style."""
    if not styles:
        raise ValueError("no styles to average")
    first = styles[0]
    if any(
        s.ref_width != first.ref_width
        or len(s.line_values) != len(first.line_values)
        or len(s.wash_values) != len(first.wash_values)
        for s in styles
    ):
        raise ValueError("styles must share ref_width and quantile counts")
    return replace(
        first,
        stroke_width=round(float(np.mean([s.stroke_width for s in styles])), 3),
        line_values=tuple(round(float(v), 3) for v in np.mean([s.line_values for s in styles], axis=0)),
        wash_values=tuple(round(float(v), 3) for v in np.mean([s.wash_values for s in styles], axis=0)),
    )