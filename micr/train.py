"""Train the MICR glyph classifier.

    python -m micr.train --data dataset --epochs 20 --version v1

Writes ``models/micr_cnn_v1.pth`` (best val accuracy) plus a training report.
The checkpoint carries its own class list, input size and normalisation, so
export.py never has to be told them again.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .classes import CLASSES, INPUT_HEIGHT, INPUT_MEAN, INPUT_STD, INPUT_WIDTH, NUM_CLASSES
from torch.utils.data import ConcatDataset

from .dataset import build_dataset, build_loader, make_loader, repeats_for_weight
from .metrics import evaluate_model, format_report, line_accuracy_note
from .model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    log_every: int = 50,
) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0
    started = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch = targets.size(0)
        running_loss += loss.item() * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += batch

        if step % log_every == 0:
            print(
                f"    step {step:>5}/{len(loader)}  "
                f"loss {running_loss / seen:.4f}  acc {correct / seen * 100:.2f}%",
                end="\r",
                flush=True,
            )

    print(" " * 80, end="\r")
    return {
        "loss": running_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MICR E-13B classifier.")
    parser.add_argument("--data", default="dataset", help="dataset root (train/ and val/)")
    parser.add_argument("--out", default="models", help="checkpoint output directory")
    parser.add_argument("--version", default="v1", help="model version tag for filenames")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--online-aug",
        default="online",
        choices=["online", "eval", "none"],
        help="per-epoch augmentation on top of the baked PNGs",
    )
    parser.add_argument(
        "--real-data",
        help="folder of real check crops (from micr.segment) to mix into training",
    )
    parser.add_argument(
        "--real-weight",
        type=float,
        default=0.25,
        help="target share of each epoch drawn from --real-data",
    )
    parser.add_argument("--patience", type=int, default=6, help="early-stop patience")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        if not (data_root / split).is_dir():
            raise SystemExit(
                f"missing {data_root / split}. Run micr.synth first to generate it."
            )

    train_set = build_dataset(data_root / "train", mode=args.online_aug)
    parts = [train_set]
    real_set = None
    repeats = 0

    if args.real_data:
        real_set = build_dataset(args.real_data, mode=args.online_aug)
        if len(real_set) == 0:
            raise SystemExit(f"{args.real_data} contains no crops")
        repeats = repeats_for_weight(len(train_set), len(real_set), args.real_weight)
        parts.extend([real_set] * repeats)

    train_dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True, workers=args.workers
    )

    val_loader, val_set = build_loader(
        data_root / "val",
        mode="eval",
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        workers=args.workers,
    )

    model = build_model(NUM_CLASSES, dropout=args.dropout).to(device)
    params = count_parameters(model)

    print(f"device:     {device}")
    print(f"model:      {model.arch}  {params:,} params (~{params * 4 / 1024:.0f} KB fp32)")
    print(f"train:      {len(train_set):,} synthetic   online aug: {args.online_aug}")
    counts = train_set.class_counts()

    if real_set is not None:
        real_counts = real_set.class_counts()
        contributed = len(real_set) * repeats
        share = contributed / (len(train_set) + contributed)
        print(
            f"real:       {len(real_set):,} crops x{repeats} = {contributed:,} "
            f"({share * 100:.0f}% of each epoch, target {args.real_weight * 100:.0f}%)"
        )
        empty = [name for name, n in real_counts.items() if n == 0]
        if empty:
            print(f"            no real samples for {empty} -- synthetic only there")
        for name, n in real_counts.items():
            counts[name] += n * repeats

    print(f"val:        {len(val_set):,} images")
    thin = {k: v for k, v in counts.items() if v < 100}
    if thin:
        print(f"WARNING: thin training classes: {thin}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Short linear warmup then cosine decay, stepped once per epoch. Written by
    # hand rather than with OneCycleLR, which divides by zero when a phase
    # rounds down to a single step (e.g. --epochs 4).
    warmup_epochs = min(2, args.epochs // 10)

    def lr_scale(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return max(0.02, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ckpt_path = out_dir / f"micr_cnn_{args.version}.pth"
    report_path = out_dir / f"micr_cnn_{args.version}_training_report.json"

    best_accuracy = -1.0
    best_epoch = -1
    epochs_without_gain = 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:>3}/{args.epochs}  lr {lr_now:.2e}")
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_stats = evaluate_model(model, val_loader, device, criterion)
        scheduler.step()

        print(
            f"    train loss {train_stats['loss']:.4f} acc {train_stats['accuracy'] * 100:6.3f}%"
            f"   |   val loss {val_stats['loss']:.4f} acc {val_stats['accuracy'] * 100:6.3f}%"
            f"   ({train_stats['seconds']:.0f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "seconds": train_stats["seconds"],
            }
        )

        if val_stats["accuracy"] > best_accuracy:
            best_accuracy = val_stats["accuracy"]
            best_epoch = epoch
            epochs_without_gain = 0
            torch.save(
                {
                    "format_version": 1,
                    "arch": model.arch,
                    "version": args.version,
                    "model_state": model.state_dict(),
                    "classes": list(CLASSES),
                    "num_classes": NUM_CLASSES,
                    "input_size": {
                        "height": INPUT_HEIGHT,
                        "width": INPUT_WIDTH,
                        "channels": 1,
                    },
                    "input_range": [0.0, 1.0],
                    "normalization": {"mean": INPUT_MEAN, "std": INPUT_STD, "baked_in": True},
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_accuracy": best_accuracy,
                    "val_confusion": val_stats["confusion"],
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "torch_version": torch.__version__,
                    "platform": platform.platform(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"    saved {ckpt_path}  (val acc {best_accuracy * 100:.3f}%)")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"    no val gain for {args.patience} epochs, stopping early")
                break

    minutes = (time.time() - started) / 60
    print(f"\nbest val accuracy {best_accuracy * 100:.3f}% at epoch {best_epoch} "
          f"({minutes:.1f} min total)")

    # Final report from the best checkpoint, not the last epoch.
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    final = evaluate_model(model, val_loader, device, criterion)
    print()
    print(format_report(final, "val (best checkpoint)"))
    print(line_accuracy_note(final))

    report_path.write_text(
        json.dumps(
            {
                "version": args.version,
                "checkpoint": str(ckpt_path),
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "parameters": params,
                "history": history,
                "final_val": final,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {report_path}")
    print(
        "\nQuote the headline accuracy from test_real, not from val -- val is our own "
        "renderer marking its own homework:\n"
        f"  python -m micr.evaluate --checkpoint {ckpt_path} --data {data_root / 'test_real'}"
    )


if __name__ == "__main__":
    main()
