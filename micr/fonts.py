"""E-13B font loading and glyph-mapping resolution.

The single biggest way to waste a day on this project is to generate 60,000
training images from a font whose symbol glyphs are on different keys than you
assumed, and end up with a model that confidently calls the transit symbol a
dash. So: probe the font's cmap, pick a mapping that actually resolves, and
render a contact sheet for a human to eyeball before any data is generated.

CLI:
    python -m micr.fonts --font path/to/E13B.ttf --preview charmap.png
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .classes import (
    ASCII_CANDIDATES,
    CLASSES,
    DIGIT_CLASSES,
    E13B_CODEPOINTS,
    SPEC_CODEPOINTS,
    UNICODE_OCR_CODEPOINTS,
    codepoints_to_chars,
    full_charmap,
)


@lru_cache(maxsize=16)
def font_codepoints(font_path: str) -> frozenset[int]:
    """Every codepoint the font actually has a glyph for."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "fontTools is required to inspect fonts. pip install fonttools"
        ) from exc

    with TTFont(font_path, fontNumber=0, lazy=True) as font:
        return frozenset(font.getBestCmap().keys())


@lru_cache(maxsize=64)
def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def candidate_charmaps() -> list[tuple[str, dict[str, str]]]:
    """Symbol mappings to try, most-likely first.

    ``e13b-verified`` leads because it is the one checked against real check
    print. ``unicode-ocr-block`` is last of the codepoint maps on purpose: it
    matches the same four codepoints but has on-us and dash swapped relative to
    the actual glyphs, so if it ever wins it is because the first two did not,
    and the contact sheet needs a careful look.
    """
    candidates: list[tuple[str, dict[str, str]]] = [
        ("e13b-verified", codepoints_to_chars(E13B_CODEPOINTS)),
        ("spec-unicode", codepoints_to_chars(SPEC_CODEPOINTS)),
    ]
    candidates.extend(ASCII_CANDIDATES)
    candidates.append(("unicode-ocr-block", codepoints_to_chars(UNICODE_OCR_CODEPOINTS)))
    return candidates


def resolve_charmap(
    font_path: str | Path,
    override: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """Return (class -> character, source name) for a font.

    ``override`` is a symbol-only or full mapping loaded from --charmap JSON and
    is trusted without probing, since a human wrote it.
    """
    font_path = str(font_path)

    if override:
        # Trusted as-is: a human wrote it. full_charmap fills in any digits the
        # override leaves out, and raises if a class is still missing.
        return full_charmap(dict(override)), "override"

    available = font_codepoints(font_path)

    missing_digits = [d for d in DIGIT_CLASSES if ord(d) not in available]
    if missing_digits:
        raise RuntimeError(
            f"{font_path} has no glyphs for digits {missing_digits}. "
            "This does not look like a usable E-13B font."
        )

    for name, symbols in candidate_charmaps():
        if all(ord(ch) in available for ch in symbols.values()):
            return full_charmap(symbols), name

    raise RuntimeError(
        f"Could not find the 4 E-13B symbols in {font_path}.\n"
        "Inspect the font's cmap and pass an explicit mapping, e.g.\n"
        '  --charmap charmap.json  with  {"transit": "a", "amount": "b", '
        '"onus": "c", "dash": "d"}\n'
        f"Font exposes {len(available)} codepoints; "
        f"non-ASCII ones: {sorted(hex(c) for c in available if c > 127)[:32]}"
    )


def render_contact_sheet(
    font_path: str | Path,
    charmap: dict[str, str],
    out_path: str | Path,
    cell: int = 96,
) -> Path:
    """Render all 14 glyphs with their class labels, for human verification."""
    cols = 7
    rows = (len(CLASSES) + cols - 1) // cols
    label_h = 22
    sheet = Image.new("L", (cols * cell, rows * (cell + label_h)), 255)
    draw = ImageDraw.Draw(sheet)
    glyph_font = load_font(str(font_path), int(cell * 0.62))

    for i, class_name in enumerate(CLASSES):
        cx, cy = (i % cols) * cell, (i // cols) * (cell + label_h)
        draw.rectangle([cx, cy, cx + cell - 1, cy + cell + label_h - 1], outline=200)

        ch = charmap[class_name]
        bbox = draw.textbbox((0, 0), ch, font=glyph_font)
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx + (cell - gw) / 2 - bbox[0], cy + (cell - gh) / 2 - bbox[1]),
            ch,
            font=glyph_font,
            fill=0,
        )
        draw.text((cx + 4, cy + cell + 4), f"{i:>2} {class_name}", fill=90)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def load_charmap_json(path: str | Path | None) -> dict[str, str] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an E-13B font.")
    parser.add_argument("--font", required=True, help="path to the E-13B .ttf/.otf")
    parser.add_argument("--charmap", help="JSON overriding the symbol mapping")
    parser.add_argument(
        "--preview",
        default="charmap_preview.png",
        help="where to write the contact sheet",
    )
    args = parser.parse_args()

    charmap, source = resolve_charmap(args.font, load_charmap_json(args.charmap))
    print(f"font:    {args.font}")
    print(f"mapping: {source}")
    for class_name in CLASSES:
        ch = charmap[class_name]
        print(f"  {class_name:<8} -> {ch!r}  (U+{ord(ch):04X})")

    out = render_contact_sheet(args.font, charmap, args.preview)
    print(f"\ncontact sheet: {out}")
    print("Check every glyph looks like real E-13B print before generating data.")


if __name__ == "__main__":
    main()
