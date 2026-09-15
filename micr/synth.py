"""Synthetic E-13B dataset generation.

Renders each of the 14 glyphs from the font file thousands of times with
randomised print and capture conditions, and writes them straight into the
``ImageFolder`` layout from spec section 7. The folder name is the label, so
nothing is ever typed by hand.

    dataset/train/<class>/<class>_00001.png
    dataset/val/<class>/<class>_00001.png

CLI:
    python -m micr.synth --font E13B.ttf --out dataset \\
        --train-per-class 4000 --val-per-class 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .augment import augment_gray, get_pipeline
from .classes import CLASSES, INPUT_HEIGHT, INPUT_WIDTH
from .fonts import load_charmap_json, load_font, render_contact_sheet, resolve_charmap

SUPERSAMPLE = 4


@dataclass
class SynthConfig:
    """Everything a worker process needs. Must stay picklable."""

    fonts: list[str]
    charmap: dict[str, str]
    width: int = INPUT_WIDTH
    height: int = INPUT_HEIGHT
    augment: bool = True
    aug_strength: float = 1.0
    neighbor_bleed_prob: float = 0.25
    fill_ratio_range: tuple[float, float] = (0.62, 0.90)
    stroke_jitter: bool = True
    # Reject crops the augmentation degraded past readability -- a blank grey
    # square filed under "9" is label noise, not hard training data. 0 disables.
    min_contrast: float = 28.0
    max_render_attempts: int = 4
    seed: int = 1234


# --- rendering primitives --------------------------------------------------


REFERENCE_GLYPH = "0"
_PROBE_SIZE = 100


@lru_cache(maxsize=512)
def _reference_metrics(font_path: str, target_h: int) -> tuple[int, int, int]:
    """(font size, reference glyph top, reference glyph height) at ``target_h``.

    Measured once per (font, height) so every glyph in the dataset is rendered
    at one consistent scale on one consistent baseline.
    """
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    probe = load_font(font_path, _PROBE_SIZE)
    probe_bbox = measure.textbbox((0, 0), REFERENCE_GLYPH, font=probe)
    probe_h = max(probe_bbox[3] - probe_bbox[1], 1)

    size = max(8, int(round(_PROBE_SIZE * target_h / probe_h)))
    final = load_font(font_path, size)
    bbox = measure.textbbox((0, 0), REFERENCE_GLYPH, font=final)
    return size, bbox[1], max(bbox[3] - bbox[1], 1)


def _render_ink_alpha(
    ch: str,
    font_path: str,
    rng: np.random.Generator,
    cfg: SynthConfig,
    x_shift: float = 0.0,
) -> np.ndarray:
    """Render one glyph and return a float alpha map (1.0 = full ink).

    Rendered at SUPERSAMPLE resolution then area-downscaled, which is what
    gives the strokes realistic soft edges instead of hard aliasing.
    """
    big_w = cfg.width * SUPERSAMPLE
    big_h = cfg.height * SUPERSAMPLE

    fill_ratio = rng.uniform(*cfg.fill_ratio_range)
    target_h = big_h * fill_ratio

    # Sizing and vertical placement are both derived from a REFERENCE glyph
    # ('0'), never from the glyph being drawn. Scaling each character to fill
    # the crop would stretch the short E-13B dash into a full-height block and
    # throw away a cue the model should be using. Every glyph therefore shares
    # one baseline and keeps its true relative height, exactly as a crop cut
    # from a fixed-height MICR band would. The height is rounded so the metric
    # cache actually hits; sub-pixel differences at 4x supersampling vanish in
    # the downscale anyway.
    size, ref_top, ref_h = _reference_metrics(font_path, round(target_h))

    font = load_font(font_path, size)
    canvas = Image.new("L", (big_w, big_h), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), ch, font=font)
    gw = bbox[2] - bbox[0]

    # A little padding plus jitter, because the runtime segmenter will not cut
    # perfectly centred crops either.
    jitter_x = rng.uniform(-0.07, 0.07) * big_w + x_shift * big_w
    jitter_y = rng.uniform(-0.07, 0.07) * big_h
    x = (big_w - gw) / 2 - bbox[0] + jitter_x
    y = (big_h - ref_h) / 2 - ref_top + jitter_y
    draw.text((x, y), ch, font=font, fill=255)

    ink = np.asarray(canvas, dtype=np.uint8)

    # Print weight: heavy toner vs worn/faint ribbon.
    if cfg.stroke_jitter:
        roll = rng.random()
        if roll < 0.3:
            k = int(rng.integers(2, 4)) * SUPERSAMPLE // 2 * 2 + 1
            ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
        elif roll < 0.55:
            k = int(rng.integers(2, 4)) * SUPERSAMPLE // 2 * 2 + 1
            ink = cv2.erode(ink, np.ones((k, k), np.uint8))

    # Slight rotation at high res (cheaper artefacts than rotating the crop).
    angle = rng.uniform(-2.5, 2.5)
    if abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((big_w / 2, big_h / 2), angle, 1.0)
        ink = cv2.warpAffine(
            ink, matrix, (big_w, big_h), flags=cv2.INTER_LINEAR, borderValue=0
        )

    small = cv2.resize(ink, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def _paper_background(rng: np.random.Generator, cfg: SynthConfig) -> np.ndarray:
    """Check stock: off-white, uneven lighting, a bit of texture."""
    base = rng.uniform(196, 255)
    bg = np.full((cfg.height, cfg.width), base, dtype=np.float32)

    # Low-frequency illumination gradient (shadow across the band).
    yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float32)
    gx = rng.uniform(-1, 1)
    gy = rng.uniform(-1, 1)
    amplitude = rng.uniform(0, 42)
    ramp = (xx / max(cfg.width - 1, 1) * gx) + (yy / max(cfg.height - 1, 1) * gy)
    bg += ramp * amplitude

    # Paper grain.
    bg += rng.normal(0, rng.uniform(1.0, 5.0), bg.shape)

    # Faint background pattern / security tint line, occasionally.
    if rng.random() < 0.2:
        row = int(rng.integers(0, cfg.height))
        thickness = int(rng.integers(1, 3))
        bg[row : row + thickness, :] -= rng.uniform(8, 30)

    return np.clip(bg, 0, 255)


def render_sample(class_name: str, rng: np.random.Generator, cfg: SynthConfig) -> np.ndarray:
    """One finished uint8 grayscale training crop, retried until it is legible."""
    img = _render_once(class_name, rng, cfg)
    if cfg.min_contrast <= 0:
        return img
    for _ in range(cfg.max_render_attempts - 1):
        low, high = np.percentile(img, (5, 95))
        if high - low >= cfg.min_contrast:
            break
        img = _render_once(class_name, rng, cfg)
    return img


def _render_once(class_name: str, rng: np.random.Generator, cfg: SynthConfig) -> np.ndarray:
    font_path = cfg.fonts[int(rng.integers(len(cfg.fonts)))]
    alpha = _render_ink_alpha(cfg.charmap[class_name], font_path, rng, cfg)

    bg = _paper_background(rng, cfg)
    ink_value = rng.uniform(0, 78)  # magnetic ink is black but cameras disagree
    img = bg * (1.0 - alpha) + ink_value * alpha

    # Slivers of the neighbouring glyphs, which imperfect segmentation leaves in.
    if rng.random() < cfg.neighbor_bleed_prob:
        for side in (-1, 1):
            if rng.random() < 0.5:
                continue
            neighbour = CLASSES[int(rng.integers(len(CLASSES)))]
            shift = side * rng.uniform(0.62, 0.85)
            n_alpha = _render_ink_alpha(
                cfg.charmap[neighbour], font_path, rng, cfg, x_shift=shift
            )
            img = img * (1.0 - n_alpha) + ink_value * n_alpha

    img = np.clip(img, 0, 255).astype(np.uint8)

    if cfg.augment:
        img = augment_gray(_pipeline_for(cfg), img)

    if img.shape != (cfg.height, cfg.width):
        img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    return img


# --- worker plumbing -------------------------------------------------------

_PIPELINE_CACHE: dict[tuple[bool, float], object] = {}


def _pipeline_for(cfg: SynthConfig):
    key = (cfg.augment, cfg.aug_strength)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = get_pipeline("bake", cfg.aug_strength)
    return _PIPELINE_CACHE[key]


def _generate_chunk(task: tuple[SynthConfig, str, str, int, int, int]) -> int:
    """Write ``count`` images for one class into ``split_dir``."""
    cfg, split_dir, class_name, start_index, count, seed = task

    # Parallelism here comes from processes, so OpenCV's own thread pool is pure
    # overhead -- and on a many-core box (an EC2 instance, say) N workers each
    # spawning M threads oversubscribes the CPU badly. The ops are on 128x192
    # buffers; single-threaded is faster anyway.
    cv2.setNumThreads(0)

    out_dir = Path(split_dir) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(count):
        img = render_sample(class_name, rng, cfg)
        path = out_dir / f"{class_name}_{start_index + i:06d}.png"
        cv2.imwrite(str(path), img)
    return count


def generate_split(
    cfg: SynthConfig,
    split_dir: Path,
    per_class: int,
    seed_offset: int,
    workers: int,
    chunk_size: int = 250,
) -> int:
    tasks: list[tuple[SynthConfig, str, str, int, int, int]] = []
    for class_idx, class_name in enumerate(CLASSES):
        remaining = per_class
        start = 0
        while remaining > 0:
            count = min(chunk_size, remaining)
            seed = cfg.seed + seed_offset + class_idx * 100_003 + start
            tasks.append((cfg, str(split_dir), class_name, start, count, seed))
            start += count
            remaining -= count

    total = 0
    done_chunks = 0
    started = time.time()

    def _progress() -> None:
        pct = 100.0 * done_chunks / max(len(tasks), 1)
        elapsed = time.time() - started
        print(
            f"  {split_dir.name}: {total:>7,}/{per_class * len(CLASSES):,} images "
            f"({pct:5.1f}%)  {elapsed:6.1f}s",
            end="\r",
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            total += _generate_chunk(task)
            done_chunks += 1
            _progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for written in pool.map(_generate_chunk, tasks, chunksize=1):
                total += written
                done_chunks += 1
                _progress()

    print()
    return total


def make_empty_split(root: Path) -> None:
    """Create the test_real skeleton so crops have somewhere to land."""
    for class_name in CLASSES:
        (root / class_name).mkdir(parents=True, exist_ok=True)
    labels = root / "labels.txt"
    if not labels.exists():
        labels.write_text(
            "# one line per check: <check id>  <MICR line using T/A/O/D substitution>\n"
            "# example:\n"
            "# chk001  O001234O T123456780T 000123456789O\n",
            encoding="utf-8",
        )


def resolve_fonts(
    font_paths: list[str], charmap_override: dict[str, str] | None, verbose: bool = True
) -> tuple[list[str], dict[str, str], str]:
    """Validate the fonts and agree on one charmap across all of them."""
    for font in font_paths:
        if not Path(font).is_file():
            raise SystemExit(f"font not found: {font}")

    charmap, source = resolve_charmap(font_paths[0], charmap_override)
    if verbose:
        print(f"charmap source: {source}")
        for class_name in CLASSES:
            ch = charmap[class_name]
            print(f"  {class_name:<8} -> {ch!r} (U+{ord(ch):04X})")

    for font in font_paths[1:]:
        other, other_source = resolve_charmap(font, charmap_override)
        if other != charmap:
            raise SystemExit(
                f"{font} resolves to a different mapping ({other_source}). "
                "Pass an explicit charmap so all fonts agree."
            )

    return [str(Path(f).resolve()) for f in font_paths], charmap, source


def generate_dataset(
    cfg: SynthConfig,
    root: Path,
    train_per_class: int,
    val_per_class: int,
    workers: int,
    charmap_source: str = "",
) -> dict:
    """Render train/ and val/, seed the test_real skeleton, write the metadata."""
    root.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(cfg.fonts[0], cfg.charmap, root / "charmap_preview.png")
    print(f"contact sheet: {root / 'charmap_preview.png'}  <- verify the glyphs")

    started = time.time()
    print(f"\ngenerating with {workers} worker(s)")
    n_train = generate_split(cfg, root / "train", train_per_class, 0, workers)
    n_val = generate_split(cfg, root / "val", val_per_class, 5_000_003, workers)
    make_empty_split(root / "test_real")

    meta = {
        "classes": list(CLASSES),
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "train_images": n_train,
        "val_images": n_val,
        "image_size": {"width": cfg.width, "height": cfg.height},
        "charmap": cfg.charmap,
        "charmap_source": charmap_source,
        "fonts": cfg.fonts,
        "augmented": cfg.augment,
        "aug_strength": cfg.aug_strength,
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    (root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    minutes = (time.time() - started) / 60
    print(f"\ntrain: {n_train:,}   val: {n_val:,}   ({minutes:.1f} min)")
    print(f"metadata: {root / 'dataset_meta.json'}")
    return meta


def default_workers(requested: int = 0) -> int:
    return requested or max(1, (os.cpu_count() or 2) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic E-13B dataset.")
    parser.add_argument(
        "--font",
        action="append",
        required=True,
        help="E-13B font file; repeat the flag to mix several fonts",
    )
    parser.add_argument("--charmap", help="JSON overriding the symbol->character mapping")
    parser.add_argument("--out", default="dataset", help="dataset root directory")
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--width", type=int, default=INPUT_WIDTH)
    parser.add_argument("--height", type=int, default=INPUT_HEIGHT)
    parser.add_argument("--aug-strength", type=float, default=1.0)
    parser.add_argument("--neighbor-bleed-prob", type=float, default=0.25)
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=28.0,
        help="reject crops whose p5-p95 range is below this; 0 disables",
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    args = parser.parse_args()

    fonts, charmap, source = resolve_fonts(args.font, load_charmap_json(args.charmap))
    workers = default_workers(args.workers)
    cfg = SynthConfig(
        fonts=fonts,
        charmap=charmap,
        width=args.width,
        height=args.height,
        augment=not args.no_augment,
        aug_strength=args.aug_strength,
        neighbor_bleed_prob=args.neighbor_bleed_prob,
        min_contrast=args.min_contrast,
        seed=args.seed,
    )

    root = Path(args.out)
    generate_dataset(
        cfg, root, args.train_per_class, args.val_per_class, workers, source
    )
    print(f"test_real skeleton + labels.txt template: {root / 'test_real'}")


if __name__ == "__main__":
    main()
