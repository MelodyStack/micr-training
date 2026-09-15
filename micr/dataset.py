"""Dataset loading on top of ``torchvision.datasets.ImageFolder``.

Two things are handled here that the stock ImageFolder does not do well for
this project:

1. **Fixed class order.** ImageFolder derives classes from whatever folders it
   finds. ``test_real`` legitimately has empty classes -- the spec's worked
   example produced no 4s, no 9s, no amount symbol, no dash -- and a missing
   folder would silently shift every index after it, so the model's "7" would
   be scored against the wrong column. ``MicrImageFolder`` always uses the
   canonical 14 classes from classes.py.
2. **Empty class folders.** Allowed, not an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from .augment import augment_gray, get_pipeline
from .classes import CLASS_TO_IDX, CLASSES, INPUT_HEIGHT, INPUT_WIDTH


def grayscale_loader(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


class AlbumentationsTransform:
    """Adapt an albumentations pipeline to the torchvision transform protocol."""

    def __init__(self, mode: str = "eval", strength: float = 1.0) -> None:
        self.mode = mode
        self.pipeline = get_pipeline(mode, strength)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = augment_gray(self.pipeline, image)
        if image.shape != (INPUT_HEIGHT, INPUT_WIDTH):
            image = np.asarray(
                Image.fromarray(image).resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
            )
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
        return tensor.unsqueeze(0)  # (1, H, W), values in [0, 1]


class MicrImageFolder(datasets.ImageFolder):
    def __init__(self, root: str | Path, transform=None) -> None:
        root = str(root)
        try:
            super().__init__(
                root, transform=transform, loader=grayscale_loader, allow_empty=True
            )
        except TypeError:
            # torchvision < 0.18 has no allow_empty
            try:
                super().__init__(root, transform=transform, loader=grayscale_loader)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"{root} has empty class folders and torchvision is too old to "
                    "allow them. Upgrade to torchvision>=0.18."
                ) from exc

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        present = {p.name for p in Path(directory).iterdir() if p.is_dir()}
        unexpected = sorted(present - set(CLASSES))
        if unexpected:
            raise RuntimeError(
                f"{directory} contains folders that are not MICR classes: {unexpected}. "
                f"Expected exactly: {list(CLASSES)}"
            )
        return list(CLASSES), dict(CLASS_TO_IDX)

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASSES}
        for _, target in self.samples:
            counts[CLASSES[target]] += 1
        return counts


def build_dataset(root: str | Path, mode: str = "eval", strength: float = 1.0) -> MicrImageFolder:
    return MicrImageFolder(root, transform=AlbumentationsTransform(mode, strength))


def _worker_init(worker_id: int) -> None:
    """Stop OpenCV oversubscribing the CPU inside DataLoader workers."""
    import cv2

    cv2.setNumThreads(0)


def make_loader(
    dataset,
    batch_size: int = 256,
    shuffle: bool = False,
    workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init if workers > 0 else None,
    )


def build_loader(
    root: str | Path,
    mode: str = "eval",
    batch_size: int = 256,
    shuffle: bool = False,
    workers: int = 0,
    strength: float = 1.0,
) -> tuple[DataLoader, MicrImageFolder]:
    dataset = build_dataset(root, mode, strength)
    return make_loader(dataset, batch_size, shuffle, workers), dataset


def repeats_for_weight(n_synthetic: int, n_real: int, weight: float) -> int:
    """How many times to repeat the real crops to hit ``weight`` of each epoch.

    A few hundred real crops next to 56,000 synthetic ones would contribute
    almost nothing to the gradient, so they are repeated. Repetition rather
    than a sampler keeps the epoch length honest and the augmentation fresh:
    every repeat draws different jitter, so the model sees variations of the
    real crop, not the identical tensor N times.
    """
    if n_real <= 0 or weight <= 0:
        return 0
    if weight >= 1.0:
        raise ValueError("real weight must be below 1.0")
    target = weight / (1.0 - weight) * n_synthetic
    return max(1, round(target / n_real))
