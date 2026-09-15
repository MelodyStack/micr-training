"""One command that turns `fonts/` + `checks/` into a ready-to-train dataset.

    python -m micr.build

You add files to two folders. Everything else is generated:

    fonts/                 the licensed E-13B .ttf            <- you
    checks/*.jpg           check photos                       <- you
    checks/labels.txt      one MICR line per photo            <- you
    ---------------------------------------------------------------------
    dataset/train/         synthetic, balanced                <- generated
    dataset/val/           synthetic, balanced                <- generated
    dataset/train_real/    crops from your checks             <- generated
    dataset/test_real/     crops from held-out checks         <- generated
    debug/                 annotated bands, one per check     <- generated

Re-run it after dropping in more checks. The synthetic set is only rebuilt when
the settings that produced it change, so the usual re-run takes seconds rather
than re-rendering 56,000 images. Check crops are always rebuilt, since that is
cheap and keeps them consistent with whatever labels.txt says right now.

    --train     continue into training once the dataset is ready
    --export    ... and on to the .tflite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .classes import CLASSES, INPUT_HEIGHT, INPUT_WIDTH
from .fonts import load_charmap_json
from .segment import (
    SegmentConfig,
    find_check_images,
    print_segment_summary,
    read_labels_file,
    segment_batch,
)
from .synth import SynthConfig, default_workers, generate_dataset, resolve_fonts

FONT_PATTERNS = ("*.ttf", "*.otf", "*.TTF", "*.OTF")

LABELS_TEMPLATE = """\
# One line per check photo in this folder:
#
#     <check id>  <MICR line>
#
# The check id is the image filename without its extension, so chk001.jpg is
# labelled by the line starting "chk001".
#
# Read the bottom band of the check left to right and substitute the symbols:
#
#     T = transit  (around the routing number)
#     A = amount   (rare on personal checks)
#     O = on-us    (around the account / check number)
#     D = dash
#
# Spaces separate fields and are ignored. Type exactly what is printed in the
# band -- including leading zeros, and NOT a routing number printed elsewhere
# on the check.
#
# Example:
# chk001  O001234O T123456780T 000123456789O
"""


def find_fonts(font_dir: Path) -> list[str]:
    found = sorted(
        {p.resolve() for pat in FONT_PATTERNS for p in font_dir.glob(pat)}
    )
    return [str(p) for p in found]


def synth_signature(cfg: SynthConfig, train_per_class: int, val_per_class: int) -> dict:
    """The settings that, if changed, invalidate the synthetic images on disk."""
    return {
        "fonts": sorted(cfg.fonts),
        "charmap": cfg.charmap,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "width": cfg.width,
        "height": cfg.height,
        "augment": cfg.augment,
        "aug_strength": cfg.aug_strength,
        "neighbor_bleed_prob": cfg.neighbor_bleed_prob,
        "min_contrast": cfg.min_contrast,
        "seed": cfg.seed,
        "classes": list(CLASSES),
    }


def existing_signature(dataset_root: Path) -> dict | None:
    meta_path = dataset_root / "dataset_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = meta["config"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    return {
        "fonts": sorted(config["fonts"]),
        "charmap": config["charmap"],
        "train_per_class": meta["train_per_class"],
        "val_per_class": meta["val_per_class"],
        "width": config["width"],
        "height": config["height"],
        "augment": config["augment"],
        "aug_strength": config["aug_strength"],
        "neighbor_bleed_prob": config["neighbor_bleed_prob"],
        "min_contrast": config["min_contrast"],
        "seed": config["seed"],
        "classes": meta["classes"],
    }


def images_present(dataset_root: Path, per_class: dict[str, int]) -> bool:
    """Cheap sanity check that the folders were not partly deleted."""
    for split, expected in per_class.items():
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            return False
        for class_name in CLASSES:
            found = sum(1 for _ in (split_dir / class_name).glob("*.png"))
            if found < expected:
                return False
    return True


def clear_split(path: Path) -> None:
    for class_name in CLASSES:
        class_dir = path / class_name
        if class_dir.is_dir():
            for png in class_dir.glob("*.png"):
                png.unlink()


def prune_debug(debug_dir: Path, check_ids: set[str]) -> int:
    """Delete debug images for checks that are no longer in checks/.

    Without this, a debug image lingers after its photo is removed or renamed,
    and the next person to open the folder is looking at a check that is not in
    the dataset.
    """
    if not debug_dir.is_dir():
        return 0
    removed = 0
    for png in debug_dir.glob("*_debug.png"):
        if png.name[: -len("_debug.png")] not in check_ids:
            png.unlink()
            removed += 1
    return removed


def run_module(module: str, extra: list[str]) -> None:
    command = [sys.executable, "-m", module, *extra]
    print(f"\n$ {' '.join(command)}\n")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the whole dataset from fonts/ and checks/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="You add fonts/ and checks/. Everything else is generated.",
    )
    parser.add_argument("--fonts-dir", default="fonts")
    parser.add_argument("--checks-dir", default="checks")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--charmap", help="JSON overriding the symbol mapping")
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--aug-strength", type=float, default=1.0)
    parser.add_argument("--min-contrast", type=float, default=28.0)
    parser.add_argument("--neighbor-bleed-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true", help="adaptive threshold")
    parser.add_argument("--pitch-min-frac", type=float, default=0.60)
    parser.add_argument(
        "--force-synth", action="store_true", help="re-render the synthetic set"
    )
    parser.add_argument("--skip-synth", action="store_true", help="checks only")
    parser.add_argument("--train", action="store_true", help="continue into training")
    parser.add_argument("--export", action="store_true", help="... and on to .tflite")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--real-weight", type=float, default=0.25)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    checks_dir = Path(args.checks_dir)
    fonts_dir = Path(args.fonts_dir)
    debug_dir = Path(args.debug_dir)

    # --- 1. synthetic ------------------------------------------------------
    print("=" * 70)
    print("[1/2] synthetic dataset")
    print("=" * 70)

    if args.skip_synth:
        print("skipped (--skip-synth)")
    else:
        fonts_dir.mkdir(parents=True, exist_ok=True)
        font_files = find_fonts(fonts_dir)
        if not font_files:
            raise SystemExit(
                f"No E-13B font in {fonts_dir}/.\n"
                "It is the one licensed asset in the project (spec section 11).\n"
                f"Drop the .ttf in {fonts_dir}/ and run this again, or pass "
                "--skip-synth to work on the check crops alone."
            )
        print(f"fonts: {[Path(f).name for f in font_files]}")

        fonts, charmap, source = resolve_fonts(
            font_files, load_charmap_json(args.charmap)
        )
        cfg = SynthConfig(
            fonts=fonts,
            charmap=charmap,
            width=INPUT_WIDTH,
            height=INPUT_HEIGHT,
            aug_strength=args.aug_strength,
            neighbor_bleed_prob=args.neighbor_bleed_prob,
            min_contrast=args.min_contrast,
            seed=args.seed,
        )

        wanted = synth_signature(cfg, args.train_per_class, args.val_per_class)
        current = existing_signature(dataset_root)
        complete = current == wanted and images_present(
            dataset_root,
            {"train": args.train_per_class, "val": args.val_per_class},
        )

        if complete and not args.force_synth:
            print(
                f"up to date: {args.train_per_class:,}/class train, "
                f"{args.val_per_class:,}/class val. "
                "Use --force-synth to re-render."
            )
        else:
            if current is not None and current != wanted:
                changed = [k for k in wanted if current.get(k) != wanted[k]]
                print(f"settings changed ({', '.join(changed)}) -- re-rendering")
            generate_dataset(
                cfg,
                dataset_root,
                args.train_per_class,
                args.val_per_class,
                default_workers(args.workers),
                source,
            )

    # --- 2. real check crops ----------------------------------------------
    print()
    print("=" * 70)
    print("[2/2] real check crops")
    print("=" * 70)

    checks_dir.mkdir(parents=True, exist_ok=True)
    labels_path = checks_dir / "labels.txt"
    if not labels_path.is_file():
        labels_path.write_text(LABELS_TEMPLATE, encoding="utf-8")
        print(f"created {labels_path}")

    jobs = find_check_images(checks_dir)
    train_real = dataset_root / "train_real"
    test_real = dataset_root / "test_real"

    stale = prune_debug(debug_dir, {check_id for _, check_id in jobs})
    if stale:
        print(f"removed {stale} debug image(s) for checks no longer in {checks_dir}/")

    if not jobs:
        print(
            f"no check photos in {checks_dir}/ yet.\n"
            f"  Drop them in, add a line per photo to {labels_path},\n"
            "  and run this again. The synthetic set above is already usable."
        )
        clear_split(train_real)
        clear_split(test_real)
        summary = None
    else:
        labels_map = read_labels_file(labels_path)
        # Always rebuilt: cheap, and it keeps the crops consistent with
        # whatever labels.txt and the segmentation code say right now.
        clear_split(train_real)
        clear_split(test_real)

        summary = segment_batch(
            jobs,
            labels_map,
            SegmentConfig(adaptive=args.adaptive, pitch_min_frac=args.pitch_min_frac),
            (INPUT_WIDTH, INPUT_HEIGHT),
            train_real,
            holdout_out=test_real,
            holdout_frac=args.holdout_frac,
            holdout_seed=args.holdout_seed,
            debug_dir=debug_dir,
        )
        print_segment_summary(summary)
        print(f"\n  debug images: {debug_dir}/")

    # --- what to do next ---------------------------------------------------
    print()
    print("=" * 70)
    has_real = bool(summary and summary["written"])
    real_args = ["--real-data", str(train_real), "--real-weight", str(args.real_weight)]
    train_args = [
        "--data", str(dataset_root),
        "--epochs", str(args.epochs),
        "--version", args.version,
        *(real_args if has_real else []),
    ]

    if summary and summary["mismatched"]:
        ids = ", ".join(cid for cid, _, _ in summary["mismatched"])
        print(f"WARNING: {len(summary['mismatched'])} check(s) produced nothing: {ids}")
        print(f"         Look at {debug_dir}/<id>_debug.png before training.")
    if summary and summary["unlabelled"]:
        ids = ", ".join(summary["unlabelled"])
        print(f"WARNING: no labels.txt line for: {ids}")

    if args.train or args.export:
        run_module("micr.train", train_args)
        if args.export:
            run_module(
                "micr.export",
                [
                    "--checkpoint", f"models/micr_cnn_{args.version}.pth",
                    "--version", args.version,
                    "--calib-data", str(dataset_root / "val"),
                ],
            )
    else:
        print("dataset ready. Next:\n")
        print(f"  python -m micr.train {' '.join(train_args)}")
        print(
            f"  python -m micr.evaluate --checkpoint models/micr_cnn_{args.version}.pth "
            f"--data {test_real}"
        )
        print(
            f"  python -m micr.export --checkpoint models/micr_cnn_{args.version}.pth "
            f"--version {args.version}"
        )
        print("\nOr re-run this with --train --export to do all three.")


if __name__ == "__main__":
    main()
