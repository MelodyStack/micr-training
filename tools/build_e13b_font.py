"""Compile the OFL E-13B SVG strokes into a TTF.

Source: https://github.com/zaxbux/MICR_E13-B_Font
        Copyright 2021 Zachary Schneider, SIL Open Font License 1.1
        Drawn from Payments Canada Standard 006 / ISO 1004:1995.

The upstream repo ships only SVG outlines, one per stroke, each in its own
tight viewBox. This builds them into a real font so Pillow can render training
glyphs from it -- no FontForge or Inkscape needed, just fontTools, which is
already a dependency.

    python tools/build_e13b_font.py --svg-dir <dir> --out fonts/E13B-OFL.ttf

Metrics come from ISO 1004: character pitch 0.125 in, character height
0.117 in. The SVG unit works out to 0.117/8.42 in, which is why the stroke
module in the path data is 0.94 units (0.0130 in, the standard stroke width).
One em is set to one character pitch so the font is naturally monospaced, which
is what E-13B is.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.misc.transform import Transform
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import SVGPath

UNITS_PER_EM = 1000

# ISO 1004 / Payments Canada Standard 006.
CHAR_HEIGHT_IN = 0.117
CHAR_PITCH_IN = 0.125
SVG_CHAR_HEIGHT = 8.42  # the digits' viewBox height upstream

# Upstream file -> (glyph name, codepoint). The four symbols sit on the Unicode
# OCR block, where U+2448 is DASH and U+2449 is CUSTOMER ACCOUNT NUMBER (on-us).
GLYPHS: list[tuple[str, str, int]] = [
    ("u0030", "zero", 0x0030),
    ("u0031", "one", 0x0031),
    ("u0032", "two", 0x0032),
    ("u0033", "three", 0x0033),
    ("u0034", "four", 0x0034),
    ("u0035", "five", 0x0035),
    ("u0036", "six", 0x0036),
    ("u0037", "seven", 0x0037),
    ("u0038", "eight", 0x0038),
    ("u0039", "nine", 0x0039),
    ("u2446", "transit", 0x2446),
    ("u2447", "amount", 0x2447),
    ("u2448", "dash", 0x2448),
    ("u2449", "onus", 0x2449),
]

VIEWBOX_RE = re.compile(r'viewBox="([\d.\-\s]+)"')


def read_viewbox(svg_path: Path) -> tuple[float, float, float, float]:
    match = VIEWBOX_RE.search(svg_path.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no viewBox in {svg_path}")
    x, y, w, h = (float(v) for v in match.group(1).split())
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an E-13B TTF from SVG strokes.")
    parser.add_argument("--svg-dir", required=True)
    parser.add_argument("--out", default="fonts/E13B-OFL.ttf")
    parser.add_argument(
        "--align",
        default="bottom",
        choices=["bottom", "center", "top"],
        help="vertical placement of glyphs shorter than the digits",
    )
    parser.add_argument("--family", default="MICR E13B OFL")
    args = parser.parse_args()

    svg_dir = Path(args.svg_dir)
    scale = UNITS_PER_EM * CHAR_HEIGHT_IN / (CHAR_PITCH_IN * SVG_CHAR_HEIGHT)
    cap_height = SVG_CHAR_HEIGHT * scale

    glyph_order = [".notdef"]
    glyphs = {}
    metrics = {}
    cmap = {}

    empty = TTGlyphPen(None)
    glyphs[".notdef"] = empty.glyph()
    metrics[".notdef"] = (UNITS_PER_EM, 0)

    print(f"scale {scale:.2f} units/svg-unit, cap height {cap_height:.0f}/{UNITS_PER_EM} em")

    for stem, name, codepoint in GLYPHS:
        svg_path = svg_dir / f"{stem}.svg"
        if not svg_path.is_file():
            raise SystemExit(f"missing {svg_path}")
        _, _, width, height = read_viewbox(svg_path)

        glyph_w = width * scale
        glyph_h = height * scale
        # Centre horizontally in the character cell; E-13B is fixed pitch, the
        # ink inside each cell is not the same width.
        dx = (UNITS_PER_EM - glyph_w) / 2.0

        if args.align == "bottom" or height == SVG_CHAR_HEIGHT:
            baseline_offset = 0.0
        elif args.align == "center":
            baseline_offset = (cap_height - glyph_h) / 2.0
        else:
            baseline_offset = cap_height - glyph_h

        # SVG y runs downward from the top of the glyph; font y runs upward
        # from the baseline. Flip, then lift to the chosen offset.
        transform = Transform(scale, 0, 0, -scale, dx, glyph_h + baseline_offset)

        # SVG carries cubic beziers; TrueType wants quadratics. 1 unit of error
        # in a 1000-unit em is far below anything a 32x48 crop can show.
        pen = TTGlyphPen(None)
        SVGPath(str(svg_path), transform=transform).draw(Cu2QuPen(pen, max_err=1.0))
        glyphs[name] = pen.glyph()
        metrics[name] = (UNITS_PER_EM, round(dx))
        cmap[codepoint] = name
        glyph_order.append(name)
        print(
            f"  U+{codepoint:04X} {name:<8} {width:>5.2f}x{height:<5.2f} svg  ->  "
            f"{glyph_w:>4.0f}x{glyph_h:<4.0f} units, baseline +{baseline_offset:.0f}"
        )

    attribution = (
        "Copyright 2021 Zachary Schneider. SIL Open Font License 1.1. "
        "Outlines from github.com/zaxbux/MICR_E13-B_Font, drawn from "
        "Payments Canada Standard 006 / ISO 1004:1995. "
        "Compiled to TTF with fontTools."
    )

    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=round(cap_height), descent=0)
    fb.setupNameTable(
        {
            "familyName": args.family,
            "styleName": "Regular",
            "uniqueFontIdentifier": f"{args.family}; fontTools build",
            "fullName": args.family,
            "psName": args.family.replace(" ", ""),
            "version": "Version 1.000",
            "copyright": attribution,
            "licenseDescription": "SIL Open Font License, Version 1.1",
            "licenseInfoURL": "https://scripts.sil.org/OFL",
        }
    )
    fb.setupOS2(
        sTypoAscender=round(cap_height),
        sTypoDescender=0,
        sCapHeight=round(cap_height),
        usWinAscent=round(cap_height),
        usWinDescent=0,
        achVendID="NONE",
    )
    fb.setupPost()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_path))
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"attribution: {attribution}")


if __name__ == "__main__":
    main()
