"""Cut the MICR band out of a real check photo and split it into glyph crops.

Spec section 9. This is deliberately the same sequence the app must run on a
camera frame -- locate band, deskew, binarize, find character boundaries -- so
writing it to build ``test_real`` also builds the runtime piece.

    # one check
    python -m micr.segment --image check.jpg --check-id chk001 \\
        --micr "O001234O T123456780T 000123456789O" --out dataset/test_real

    # a directory, labelled from test_real/labels.txt
    python -m micr.segment --checks photos/ --labels dataset/test_real/labels.txt \\
        --out dataset/test_real

Crops are taken at the **full band height**, not the glyph's own bounding box.
That matters: the synthetic renderer places every glyph on one shared baseline
at its true relative height, so a dash is short within its crop. Cropping tight
to each glyph here would rescale the dash to full height and hand the model
test data that does not look like what it trained on.

Nothing is written unless the glyph count matches the label string exactly. A
segmenter that finds 31 glyphs in a 32-glyph line would otherwise file every
crop after the missed one under the wrong class, which is worse than no data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES, SUBSTITUTION, micr_line_to_labels

# --- tuning ----------------------------------------------------------------


class SegmentationError(Exception):
    """One check could not be segmented. Never fatal to a batch."""


@dataclass
class SegmentConfig:
    search_frac: float = 0.42  # fraction of check height, measured from bottom
    min_band_width_frac: float = 0.30  # band must span this much of the width
    max_band_height_frac: float = 0.16  # ... and be no taller than this
    band_pad_frac: float = 0.22  # padding added above/below the band
    max_deskew_deg: float = 12.0
    pitch_min_frac: float = 0.60  # centre hops below this * width are intra-glyph
    max_neighbour_pitch: float = 3.0  # lone marks further out than this are not glyphs
    min_glyph_width_frac: float = 0.12  # of glyph width; below is a speck
    crop_pad_frac: float = 0.18  # side padding on each glyph crop
    adaptive: bool = False
    # Finding the check inside a phone photo, then working out which way is up.
    detect_document: bool = True
    doc_work_width: float = 1000.0  # detection runs downscaled, for speed
    doc_min_area_frac: float = 0.12  # the sheet must fill this much of the frame
    auto_orient: bool = True
    min_plausible_glyphs: int = 8  # fewer than this is not a MICR line
    max_plausible_glyphs: int = 50  # ... and more than this is texture, not print
    max_band_candidates: int = 6  # strips to score before picking one


# --- stages ----------------------------------------------------------------


def load_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SegmentationError(f"could not read {path}")
    return image


def binarize(gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Return ink=255, paper=0."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if cfg.adaptive:
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def find_document(gray: np.ndarray, cfg: SegmentConfig) -> np.ndarray | None:
    """Corners of the check within the photo, or None if it fills the frame.

    A phone photo is mostly desk. Locating the band inside the full frame means
    competing with shadows, table edges and whatever else is lying around, and
    the band ends up a small fraction of the image height. Finding the sheet
    first and rectifying it removes the background and the perspective in one
    step -- which is also what the app's capture overlay has to do.
    """
    height, width = gray.shape
    scale = cfg.doc_work_width / max(width, 1)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(contour)
        if area < frame_area * cfg.doc_min_area_frac:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32) / scale

    # No clean quad: fall back to the bounding box of the largest blob, which
    # still crops most of the desk away.
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < frame_area * cfg.doc_min_area_frac:
        return None
    x, y, w, h = cv2.boundingRect(biggest)
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return box / scale


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    ordered[0] = quad[np.argmin(total)]
    ordered[2] = quad[np.argmax(total)]
    ordered[1] = quad[np.argmin(diff)]
    ordered[3] = quad[np.argmax(diff)]
    return ordered


def warp_document(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    corners = order_corners(quad)
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return gray
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC)


def band_quality(boxes: list[tuple[int, int]], cfg: SegmentConfig) -> float:
    """How much does this look like a real MICR line rather than texture?

    Counting boxes alone is not enough -- it rewards noise, and an upside-down
    check scored higher than the right way up because a band of texture
    shattered into 70 fragments. A genuine E-13B line has a bounded number of
    characters, widths that cluster (the glyphs differ, but only within about
    2x), and centres that sit on a constant pitch.
    """
    count = len(boxes)
    if count < cfg.min_plausible_glyphs or count > cfg.max_plausible_glyphs:
        return 0.0

    centres = np.array([(a + b) / 2.0 for a, b in boxes], dtype=float)
    widths = np.array([b - a for a, b in boxes], dtype=float)
    if widths.mean() <= 0:
        return 0.0

    deltas = np.diff(centres)
    pitch = float(np.median(deltas))
    if pitch <= 0:
        return 0.0

    origin = float(centres[0])
    for _ in range(3):
        cells = np.round((centres - origin) / pitch)
        origin += float((centres - origin - cells * pitch).mean())
    residual = (centres - origin) - np.round((centres - origin) / pitch) * pitch
    grid_fit = max(0.0, 1.0 - 2.0 * float(np.sqrt((residual**2).mean())) / pitch)

    width_cv = float(widths.std() / widths.mean())
    width_fit = max(0.0, 1.0 - width_cv)

    return count * grid_fit * width_fit


def score_orientation(gray: np.ndarray, cfg: SegmentConfig) -> float:
    """Score this rotation as "check with a MICR band along the bottom"."""
    try:
        return read_band(gray, cfg)[0]
    except Exception:
        return 0.0


def orient(gray: np.ndarray, cfg: SegmentConfig) -> tuple[np.ndarray, int]:
    """Rotate so the MICR band sits along the bottom. Returns (image, degrees)."""
    best_image, best_score, best_k = gray, -1.0, 0
    for k in range(4):
        candidate = np.ascontiguousarray(np.rot90(gray, k))
        score = score_orientation(candidate, cfg)
        if score > best_score:
            best_image, best_score, best_k = candidate, score, k
    return best_image, best_k * 90


def band_candidates(gray: np.ndarray, cfg: SegmentConfig) -> list[tuple[int, int]]:
    """Candidate (top, bottom) row ranges that might be the MICR band.

    Returned most-ink-first, but ink is a weak signal on its own: a signature
    line, a memo line or a printed "CONTROLLERS WARRANT" caption all carry more
    ink than the MICR line, and picking the heaviest strip grabbed exactly those
    on real checks. The caller scores each candidate by how much it actually
    parses as E-13B and takes the winner.
    """
    height, width = gray.shape
    search_top = int(height * (1.0 - cfg.search_frac))
    strip = gray[search_top:, :]

    binary = binarize(strip, cfg)

    # Smear horizontally so a line of separate glyphs becomes one blob, and
    # the decorative border stays a thin line rather than joining in.
    kernel_w = max(15, width // 40)
    smeared = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w < width * cfg.min_band_width_frac:
            continue
        if h > height * cfg.max_band_height_frac or h < 4:
            continue
        candidates.append((area, y, h))

    ranges = []
    for _, y, h in sorted(candidates, key=lambda c: c[0], reverse=True)[
        : cfg.max_band_candidates
    ]:
        pad = int(h * cfg.band_pad_frac)
        ranges.append(
            (max(0, search_top + y - pad), min(height, search_top + y + h + pad))
        )
    return ranges


def read_band(gray: np.ndarray, cfg: SegmentConfig):
    """Pick the best-scoring candidate band and segment it.

    Returns (score, (top, bottom), band image, angle, boxes).
    """
    best = None
    for top, bottom in band_candidates(gray, cfg):
        band, angle = deskew(gray[top:bottom, :], cfg)
        boxes = find_glyph_boxes(binarize(band, cfg), cfg)
        score = band_quality(boxes, cfg)
        if best is None or score > best[0]:
            best = (score, (top, bottom), band, angle, boxes)
    if best is None:
        raise SegmentationError(
            "no candidate band in the bottom strip. Try --adaptive, or "
            "--search-frac to widen the area searched."
        )
    return best


def locate_band(gray: np.ndarray, cfg: SegmentConfig) -> tuple[int, int]:
    """Row range of the best-scoring MICR band."""
    return read_band(gray, cfg)[1]


def deskew(band: np.ndarray, cfg: SegmentConfig) -> tuple[np.ndarray, float]:
    """Rotate the band so the text baseline is horizontal."""
    binary = binarize(band, cfg)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 20:
        return band, 0.0

    angle = cv2.minAreaRect(points)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) > cfg.max_deskew_deg:
        # A wild angle means minAreaRect locked onto something that is not the
        # text; leaving the band alone beats rotating it into nonsense.
        return band, 0.0
    if abs(angle) < 0.1:
        return band, 0.0

    height, width = band.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        band,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle)


def _ink_runs(ink: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start = None
    for x, filled in enumerate(ink):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            runs.append([start, x])
            start = None
    if start is not None:
        runs.append([start, len(ink)])
    return runs


def _estimate_pitch(
    runs: list[list[int]], centres: np.ndarray, cfg: SegmentConfig
) -> tuple[float, float]:
    """(typical run width, character pitch).

    The pitch is the median spacing between run centres, ignoring the short
    hops between strokes inside one symbol. With only a handful of multi-stroke
    symbols in a 30-odd glyph line, the width median still lands on a digit.
    """
    base = float(np.median([r[1] - r[0] for r in runs]))
    deltas = np.diff(centres)
    if deltas.size == 0:
        return base, 0.0
    between_glyphs = deltas[deltas >= base * cfg.pitch_min_frac]
    return base, float(np.median(between_glyphs if between_glyphs.size else deltas))


def find_glyph_boxes(binary: np.ndarray, cfg: SegmentConfig) -> list[tuple[int, int]]:
    """Return (x_start, x_end) per glyph from the vertical projection profile.

    Runs of ink are assigned to cells of a fixed-pitch character grid rather
    than merged by gap size. Gap-based merging cannot win here: the transit and
    on-us symbols are drawn as several separate vertical strokes, so any
    threshold loose enough to join one symbol's strokes also joins two adjacent
    digits, and any threshold tight enough to separate digits shatters the
    symbols. Both failure modes were observed on a real check.

    E-13B is fixed-pitch, which resolves it. Estimate the pitch, fit a grid,
    and let cell membership decide: strokes of one symbol share a cell, adjacent
    characters do not, and the blank cells between fields simply hold no ink.

    Runs touching the image edge are dropped first -- that is the check's
    printed border, not a glyph.
    """
    projection = (binary > 0).sum(axis=0)
    ink = projection > 0
    if not ink.any():
        return []

    width = len(ink)
    runs = [r for r in _ink_runs(ink) if r[0] > 0 and r[1] < width]
    if not runs:
        return []
    if len(runs) < 2:
        return [(r[0], r[1]) for r in runs]

    centres = np.array([(a + b) / 2.0 for a, b in runs], dtype=float)
    base, pitch = _estimate_pitch(runs, centres, cfg)
    if not np.isfinite(pitch) or pitch <= 1.0:
        return [(a, b) for a, b in runs]

    # Drop isolated marks. A check's printed border rules and corner specks sit
    # a long way from the line -- on the sample check, 266 px and 496 px out,
    # against a 24 px pitch -- while every real glyph has a neighbour within a
    # couple of pitches even across the blank cells between fields. Left in,
    # they both inflate the glyph count and drag the grid origin off.
    if len(runs) >= 3:
        gaps = np.diff(centres)
        nearest = np.minimum(np.append(gaps, np.inf), np.insert(gaps, 0, np.inf))
        keep = nearest <= pitch * cfg.max_neighbour_pitch
        if 2 <= int(keep.sum()) < len(runs):
            runs = [run for run, good in zip(runs, keep) if good]
            centres = centres[keep]
            base, pitch = _estimate_pitch(runs, centres, cfg)

    # Fit the grid origin: round to cells, then shift by the mean residual so
    # the cell boundaries fall in the gaps rather than through the ink.
    origin = float(centres[0])
    for _ in range(3):
        cells = np.round((centres - origin) / pitch)
        origin += float((centres - origin - cells * pitch).mean())
    cells = np.round((centres - origin) / pitch).astype(int)

    boxes: list[tuple[int, int]] = []
    for cell in sorted(set(cells.tolist())):
        members = [runs[i] for i in range(len(runs)) if cells[i] == cell]
        boxes.append((min(m[0] for m in members), max(m[1] for m in members)))

    min_width = max(1, base * cfg.min_glyph_width_frac)
    return [(a, b) for a, b in boxes if (b - a) >= min_width]


def crop_glyphs(
    band: np.ndarray, boxes: list[tuple[int, int]], cfg: SegmentConfig, size: tuple[int, int]
) -> list[np.ndarray]:
    """Full-band-height crops, one per box, resized to (width, height)."""
    out_w, out_h = size
    height, width = band.shape
    crops = []
    for x0, x1 in boxes:
        pad = int((x1 - x0) * cfg.crop_pad_frac)
        a = max(0, x0 - pad)
        b = min(width, x1 + pad)
        crop = band[:, a:b]
        crops.append(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA))
    return crops


# --- driver ----------------------------------------------------------------


@dataclass
class SegmentResult:
    check_id: str
    boxes: list[tuple[int, int]]
    crops: list[np.ndarray]
    band: np.ndarray
    band_rows: tuple[int, int]
    angle: float
    rotation: int = 0


def segment_check(
    path: str | Path,
    check_id: str,
    cfg: SegmentConfig,
    size: tuple[int, int],
) -> SegmentResult:
    gray = load_gray(path)

    rotation = 0
    if cfg.detect_document:
        quad = find_document(gray, cfg)
        if quad is not None:
            gray = warp_document(gray, quad)
    if cfg.auto_orient:
        gray, rotation = orient(gray, cfg)

    _, rows, band, angle, boxes = read_band(gray, cfg)
    crops = crop_glyphs(band, boxes, cfg, size)
    return SegmentResult(check_id, boxes, crops, band, rows, angle, rotation)


def write_debug(result: SegmentResult, out_path: Path, labels: list[str] | None) -> None:
    canvas = cv2.cvtColor(result.band, cv2.COLOR_GRAY2BGR)
    height = canvas.shape[0]
    for i, (x0, x1) in enumerate(result.boxes):
        cv2.rectangle(canvas, (x0, 0), (x1 - 1, height - 1), (0, 140, 255), 1)
        text = SUBSTITUTION[labels[i]] if labels and i < len(labels) else str(i)
        cv2.putText(
            canvas, text, (x0, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1
        )
    scale = max(1, 1400 // max(canvas.shape[1], 1))
    if scale > 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def holdout_bucket(check_id: str, frac: float, seed: int) -> bool:
    """True if this check belongs in the held-out split.

    Split by check, never by crop. Two glyphs cut from the same photo share its
    paper, print run, lighting and camera; letting one land in train and the
    other in test leaks, and the reported accuracy comes out flattering and
    wrong. Hashed rather than shuffled so the assignment is stable when checks
    are added later.
    """
    digest = hashlib.md5(f"{seed}:{check_id}".encode()).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < frac


def read_labels_file(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(
                f"malformed labels line: {line!r}\n"
                "Expected: <check id>  <MICR line>, e.g.\n"
                "    chk001  O001234O T123456780T 000123456789O"
            )
        check_id, micr = parts
        # A filename with spaces truncates the id at the first space and silently
        # matches nothing, so reject anything the MICR alphabet cannot contain.
        stray = set(micr) - set("0123456789TAOD \t")
        if stray:
            raise SystemExit(
                f"labels line for {check_id!r} contains {sorted(stray)}, which are not "
                f"MICR characters:\n    {line}\n"
                "Check ids cannot contain spaces -- rename the photo, or use "
                "`python -m micr.ingest` to copy the photos in under clean ids."
            )
        mapping[check_id] = micr
    return mapping


CHECK_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def find_check_images(checks_dir: str | Path) -> list[tuple[Path, str]]:
    """[(path, check id)] for every photo in a folder; id is the filename stem."""
    found = sorted(p for pat in CHECK_PATTERNS for p in Path(checks_dir).glob(pat))
    return [(p, p.stem) for p in found]


def segment_batch(
    jobs: list[tuple[Path, str]],
    labels_map: dict[str, str],
    cfg: SegmentConfig,
    size: tuple[int, int],
    out_root: Path,
    holdout_out: Path | None = None,
    holdout_frac: float = 0.3,
    holdout_seed: int = 0,
    debug_dir: Path | None = None,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict:
    """Segment every job, file the crops, and return a summary dict."""
    written = 0
    unlabelled: list[str] = []
    mismatched: list[tuple[str, int, int]] = []
    failed: list[tuple[str, str]] = []
    per_class: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}

    def say(message: str) -> None:
        if not quiet:
            print(message)

    for path, check_id in jobs:
        # One unreadable photo must not take the batch down with it. At 100+
        # checks a hard failure two thirds of the way through throws away every
        # crop cut so far.
        try:
            result = segment_check(path, check_id, cfg, size)
        except Exception as exc:
            say(f"  {check_id}: FAILED -- {exc}")
            failed.append((check_id, str(exc)))
            continue

        micr = labels_map.get(check_id)

        if micr is None:
            say(
                f"  {check_id}: {len(result.boxes)} glyphs found, NO LABEL -- add a "
                f"line to labels.txt:\n      {check_id}  <MICR line>"
            )
            unlabelled.append(check_id)
            if debug_dir:
                write_debug(result, Path(debug_dir) / f"{check_id}_debug.png", None)
            continue

        labels = micr_line_to_labels(micr)
        aligned = len(labels) == len(result.boxes)
        if debug_dir:
            write_debug(
                result,
                Path(debug_dir) / f"{check_id}_debug.png",
                labels if aligned else None,
            )

        if not aligned:
            say(
                f"  {check_id}: MISMATCH -- segmented {len(result.boxes)} glyphs, "
                f"label has {len(labels)}. Nothing written for this check.\n"
                f"      Look at the debug image, then try --adaptive or "
                f"--pitch-min-frac (currently {cfg.pitch_min_frac})."
            )
            mismatched.append((check_id, len(result.boxes), len(labels)))
            continue

        destination = out_root
        if holdout_out is not None and holdout_bucket(check_id, holdout_frac, holdout_seed):
            destination = Path(holdout_out)

        angle_note = f", deskewed {result.angle:+.2f}deg" if result.angle else ""
        split_note = f" -> {destination}" if holdout_out is not None else ""
        say(f"  {check_id}: {len(labels)} glyphs{angle_note}{split_note}")

        if dry_run:
            continue

        key = str(destination)
        per_class.setdefault(key, {name: 0 for name in CLASSES})
        split_counts[key] = split_counts.get(key, 0) + 1

        for index, (label, crop) in enumerate(zip(labels, result.crops)):
            target = destination / label
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{check_id}_pos{index:02d}.png"), crop)
            per_class[key][label] += 1
            written += 1

    return {
        "checks": len(jobs),
        "written": written,
        "unlabelled": unlabelled,
        "mismatched": mismatched,
        "failed": failed,
        "per_class": per_class,
        "split_counts": split_counts,
    }


def print_segment_summary(summary: dict) -> None:
    skipped = len(summary["unlabelled"]) + len(summary["mismatched"])
    print(f"\n{summary['written']:,} crops written, {skipped} check(s) skipped")
    for key, counts in summary["per_class"].items():
        present = {k: v for k, v in counts.items() if v}
        missing = [k for k, v in counts.items() if not v]
        n_checks = summary["split_counts"][key]
        print(f"\n  {key}  ({n_checks} check(s), {sum(counts.values()):,} crops)")
        print(f"    per class: {json.dumps(present)}")
        if missing:
            print(f"    no samples for: {missing}")
    if any(any(v == 0 for v in c.values()) for c in summary["per_class"].values()):
        print(
            "\n  Rare classes need business and payroll checks -- the amount symbol "
            "barely appears on personal ones. Synthetic data covers them meanwhile."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment check photos into glyph crops.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="a single check photo")
    source.add_argument("--checks", help="directory of check photos")
    parser.add_argument("--check-id", help="id for --image; default is the filename stem")
    parser.add_argument("--micr", help="MICR line for --image, e.g. 'O013708O T1130...T ...'")
    parser.add_argument(
        "--labels",
        help="labels.txt mapping check id -> MICR line; "
        "defaults to labels.txt inside --checks",
    )
    parser.add_argument("--out", default="dataset/test_real")
    parser.add_argument(
        "--holdout-out",
        help="second split; whole checks are routed here at --holdout-frac. "
        "Use it to keep evaluation checks out of training.",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--debug-dir", help="write annotated band images here")
    parser.add_argument("--dry-run", action="store_true", help="segment but write no crops")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--adaptive", action="store_true", help="adaptive threshold")
    parser.add_argument("--search-frac", type=float, default=0.42)
    parser.add_argument("--pitch-min-frac", type=float, default=0.60)
    args = parser.parse_args()

    cfg = SegmentConfig(
        search_frac=args.search_frac,
        pitch_min_frac=args.pitch_min_frac,
        adaptive=args.adaptive,
    )
    size = (args.width, args.height)
    out_root = Path(args.out)

    if args.image:
        jobs = [(Path(args.image), args.check_id or Path(args.image).stem)]
        labels_map = {}
        if args.micr:
            labels_map[jobs[0][1]] = args.micr
        elif args.labels:
            labels_map = read_labels_file(args.labels)
    else:
        labels_path = Path(args.labels) if args.labels else Path(args.checks) / "labels.txt"
        if not labels_path.is_file():
            raise SystemExit(
                f"no labels file at {labels_path}. Put one line per check in it:\n"
                "    chk001  O001234O T123456780T 000123456789O"
            )
        labels_map = read_labels_file(labels_path)
        jobs = find_check_images(args.checks)
        if not jobs:
            raise SystemExit(f"no check images under {args.checks}")

    summary = segment_batch(
        jobs,
        labels_map,
        cfg,
        size,
        out_root,
        holdout_out=Path(args.holdout_out) if args.holdout_out else None,
        holdout_frac=args.holdout_frac,
        holdout_seed=args.holdout_seed,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        dry_run=args.dry_run,
    )
    print_segment_summary(summary)


if __name__ == "__main__":
    main()
