"""Score a trained checkpoint against a dataset split.

    python -m micr.evaluate --checkpoint models/micr_cnn_v1.pth --data dataset/test_real

Per spec section 7, the number quoted to the client comes from ``test_real``
(crops from real check photos), not from ``val``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .classes import CLASSES
from .dataset import build_loader
from .metrics import evaluate_model, format_report, line_accuracy_note
from .model import build_model


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", list(CLASSES))
    if list(classes) != list(CLASSES):
        raise SystemExit(
            f"checkpoint class order {classes} does not match classes.py {list(CLASSES)}. "
            "Retrain, or the label indices will be wrong."
        )
    model = build_model(len(classes), dropout=checkpoint.get("dropout", 0.3))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MICR checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="a split dir, e.g. dataset/test_real")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", help="also write the full report here")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    loader, dataset = build_loader(
        args.data, mode="eval", batch_size=args.batch_size, workers=args.workers
    )
    if len(dataset) == 0:
        raise SystemExit(f"{args.data} contains no images.")

    print(f"checkpoint: {args.checkpoint}  (trained {checkpoint.get('trained_at', '?')})")
    print(f"data:       {args.data}  ({len(dataset):,} crops)")
    print()

    result = evaluate_model(model, loader, device)
    print(format_report(result, Path(args.data).name))
    print(line_accuracy_note(result))

    empty = [name for name, stats in result["per_class"].items() if stats["support"] == 0]
    if empty:
        print(
            f"\nNOTE: no samples for {empty}. Rare classes (the amount symbol in "
            "particular) barely appear on personal checks -- collect business and "
            "payroll checks before trusting these columns."
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"checkpoint": args.checkpoint, "data": args.data, **result}, indent=2),
            encoding="utf-8",
        )
        print(f"\nreport: {args.json}")


if __name__ == "__main__":
    main()
