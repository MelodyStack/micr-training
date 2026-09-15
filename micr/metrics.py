"""Evaluation helpers shared by train.py and evaluate.py."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classes import CLASSES, NUM_CLASSES


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    """Return loss, accuracy, confusion matrix and per-class accuracy."""
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    seen = 0
    low_confidence = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, targets).item() * targets.size(0)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        low_confidence += int((conf < 0.9).sum().item())
        seen += targets.size(0)
        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    correct = int(np.trace(confusion))
    support = confusion.sum(axis=1)
    per_class = {
        CLASSES[i]: {
            "support": int(support[i]),
            "accuracy": float(confusion[i, i] / support[i]) if support[i] else None,
        }
        for i in range(NUM_CLASSES)
    }

    return {
        "loss": total_loss / seen if (criterion is not None and seen) else None,
        "accuracy": correct / seen if seen else 0.0,
        "correct": correct,
        "total": seen,
        "low_confidence": low_confidence,
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def top_confusions(confusion: list[list[int]], limit: int = 10) -> list[dict]:
    matrix = np.asarray(confusion)
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                pairs.append(
                    {"true": CLASSES[i], "predicted": CLASSES[j], "count": int(matrix[i, j])}
                )
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:limit]


def format_report(result: dict, title: str = "results") -> str:
    lines = [
        f"{title}: {result['correct']:,}/{result['total']:,} "
        f"= {result['accuracy'] * 100:.3f}% glyph accuracy"
    ]
    if result.get("total"):
        lines.append(
            f"  low-confidence (<0.9) predictions: {result['low_confidence']:,} "
            f"({100.0 * result['low_confidence'] / result['total']:.2f}%)"
        )
    lines.append("")
    lines.append(f"  {'class':<8} {'support':>8} {'accuracy':>10}")
    for name, stats in result["per_class"].items():
        acc = "  n/a" if stats["accuracy"] is None else f"{stats['accuracy'] * 100:8.2f}%"
        lines.append(f"  {name:<8} {stats['support']:>8,} {acc:>10}")

    confusions = top_confusions(result["confusion"])
    if confusions:
        lines.append("")
        lines.append("  top confusions (true -> predicted):")
        for c in confusions:
            lines.append(f"    {c['true']:<8} -> {c['predicted']:<8} {c['count']:>6,}")
    return "\n".join(lines)


def line_accuracy_note(result: dict) -> str:
    """A MICR line is ~30 glyphs; per-glyph accuracy compounds fast."""
    acc = result["accuracy"]
    if acc <= 0:
        return ""
    line = acc**30
    return (
        f"  at {acc * 100:.3f}% per glyph, a clean 30-glyph read succeeds "
        f"{line * 100:.1f}% of the time before multi-frame voting"
    )
