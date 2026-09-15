"""Augmentation pipelines.

Two pipelines, both on uint8 grayscale HxW arrays:

* ``build_bake_pipeline`` -- heavy. Applied once per image when the synthetic
  dataset is written to disk (spec section 7: "rendered then augmented").
* ``build_online_pipeline`` -- light. Applied per epoch during training so the
  model does not memorise the exact 4,000 frozen variants per class.

The transforms target the failure modes the spec calls out as the real
difficulty: blur, angle, shadow, folded paper, low contrast.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

from .classes import INPUT_HEIGHT, INPUT_WIDTH

# cv2.BORDER_REPLICATE. Reflect would mirror ink back into the crop and invent
# glyph fragments that no real check has; replicating the edge pixels just
# extends the paper, which is what a real crop looks like.
BORDER_REPLICATE = 1


def augment_gray(pipeline: A.Compose, image: np.ndarray) -> np.ndarray:
    """Run a pipeline on a uint8 HxW grayscale image.

    Several albumentations transforms (ISONoise, RandomShadow, ...) work in a
    colour space and reject single-channel input, so the image is replicated to
    3 channels for the pipeline and channel 0 is taken back afterwards. Taking
    one channel rather than averaging keeps the injected noise at full
    amplitude, which is the point of adding it.
    """
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    out = pipeline(image=image)["image"]
    if out.ndim == 3:
        out = out[:, :, 0]
    return np.ascontiguousarray(out)


def build_bake_pipeline(strength: float = 1.0) -> A.Compose:
    """Heavy, capture-realistic augmentation baked into the saved PNGs."""
    return A.Compose(
        [
            # Camera angle and hand shake.
            A.Affine(
                scale=(1.0 - 0.10 * strength, 1.0 + 0.10 * strength),
                translate_percent=(-0.06 * strength, 0.06 * strength),
                rotate=(-6 * strength, 6 * strength),
                shear=(-7 * strength, 7 * strength),
                border_mode=BORDER_REPLICATE,
                p=0.9,
            ),
            # Folded / curled paper. Kept deliberately gentle: the whole crop is
            # one ~2 mm glyph, so a warp strong enough to look dramatic here is
            # one that shreds the character's topology and mislabels the sample.
            A.OneOf(
                [
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.08 * strength,
                        border_mode=BORDER_REPLICATE,
                    ),
                    A.ElasticTransform(
                        alpha=8 * strength,
                        sigma=5,
                        approximate=True,
                        border_mode=BORDER_REPLICATE,
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.08 * strength, border_mode=BORDER_REPLICATE
                    ),
                ],
                p=0.3,
            ),
            # Focus and motion.
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Defocus(radius=(1, 3)),
                ],
                p=0.55,
            ),
            # Low light, shadow across the band, faded toner.
            A.RandomBrightnessContrast(
                brightness_limit=0.32 * strength,
                contrast_limit=0.34 * strength,
                p=0.8,
            ),
            A.RandomGamma(gamma_limit=(65, 145), p=0.35),
            A.OneOf(
                [
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2)),
                    A.RandomToneCurve(scale=0.3),
                ],
                p=0.3,
            ),
            # Sensor noise and small-crop resampling.
            A.GaussNoise(std_range=(0.02, 0.10 * strength + 0.02), p=0.6),
            A.ISONoise(intensity=(0.1, 0.5), p=0.2),
            A.Downscale(scale_range=(0.45, 0.9), p=0.3),
            A.ImageCompression(quality_range=(28, 92), p=0.5),
            # Ink dropout / print voids on worn checks.
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(2, 5),
                hole_width_range=(2, 5),
                fill=255,
                p=0.18,
            ),
            A.Resize(INPUT_HEIGHT, INPUT_WIDTH),
        ]
    )


def build_online_pipeline() -> A.Compose:
    """Light per-epoch jitter on top of the already-augmented PNGs."""
    return A.Compose(
        [
            A.Affine(
                scale=(0.94, 1.06),
                translate_percent=(-0.05, 0.05),
                rotate=(-4, 4),
                shear=(-4, 4),
                border_mode=BORDER_REPLICATE,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 3)), A.MotionBlur(blur_limit=3)], p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Resize(INPUT_HEIGHT, INPUT_WIDTH),
        ]
    )


def build_eval_pipeline() -> A.Compose:
    """Deterministic: resize only. Used for val, test_real and calibration."""
    return A.Compose([A.Resize(INPUT_HEIGHT, INPUT_WIDTH)])


def get_pipeline(mode: str, strength: float = 1.0) -> A.Compose:
    if mode == "bake":
        return build_bake_pipeline(strength)
    if mode in ("online", "light"):
        return build_online_pipeline()
    if mode in ("eval", "none"):
        return build_eval_pipeline()
    raise ValueError(f"unknown augmentation mode: {mode!r}")
