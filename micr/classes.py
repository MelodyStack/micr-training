"""Canonical MICR E-13B class definitions.

Single source of truth for the 14 classes. Everything downstream (dataset
folders, model output order, the labels file shipped with the .tflite) is
derived from CLASSES, so the index order can never drift between training
and the mobile app.

CLASSES is sorted because ``torchvision.datasets.ImageFolder`` sorts folder
names -- keeping the same order here means index N always means the same glyph.
"""

from __future__ import annotations

DIGIT_CLASSES = tuple(str(d) for d in range(10))
SYMBOL_CLASSES = ("amount", "dash", "onus", "transit")

# Sorted => identical to ImageFolder's class_to_idx ordering:
#   0 1 2 3 4 5 6 7 8 9 amount dash onus transit
CLASSES: tuple[str, ...] = tuple(sorted(DIGIT_CLASSES + SYMBOL_CLASSES))
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Export substitution scheme from the spec (section 8):
# T = transit, A = amount, O = on-us, D = dash.
SUBSTITUTION = {
    **{d: d for d in DIGIT_CLASSES},
    "transit": "T",
    "amount": "A",
    "onus": "O",
    "dash": "D",
}

# Crop geometry. Spec section 7: "roughly 32 x 48 px", grayscale.
INPUT_WIDTH = 32
INPUT_HEIGHT = 48
INPUT_CHANNELS = 1

# Normalisation is baked into the model (see model.py), so the app feeds
# plain [0, 1] grayscale and does not have to reproduce these constants.
INPUT_MEAN = 0.5
INPUT_STD = 0.5

# --- Font codepoints -------------------------------------------------------
#
# VERIFIED mapping, and the one to trust. Checked glyph by glyph against the
# MICR band of a real check (chk001) using the OFL E-13B font built from
# Payments Canada Standard 006 outlines:
#
#   U+2446  bar + two blocks          == the real transit symbol
#   U+2448  two bars + block          == the real on-us symbol
#   U+2449  three short bars, centred == dash (by elimination)
#
# The on-us identification is corroborated by proportion: on the real check the
# on-us symbol is 0.78x the height of a digit, and U+2448 is 0.775x. U+2447 is
# amount, where the spec and Unicode already agree and the remaining slot is
# forced.
#
# Unicode's *names* for this block are misleading here: it calls U+2448 "OCR
# DASH" and U+2449 "OCR CUSTOMER ACCOUNT NUMBER", which is on-us and dash
# swapped relative to what the glyphs actually are. Trusting those names
# produces a model that confidently reports "dash" for every on-us delimiter --
# i.e. on every check. The spec's section 4 table has U+2448 as on-us and is
# right about it; only its dash codepoint (U+2444, OCR BELT BUCKLE) is off.
E13B_CODEPOINTS = {
    "transit": 0x2446,
    "amount": 0x2447,
    "onus": 0x2448,
    "dash": 0x2449,
}

# As written in spec section 4. Kept as a fallback for fonts that really do put
# the dash on U+2444; note its other three entries agree with the verified map.
SPEC_CODEPOINTS = {
    "transit": 0x2446,
    "amount": 0x2447,
    "onus": 0x2448,
    "dash": 0x2444,
}

# Unicode's own names for the OCR block. Kept last, and only as a desperate
# fallback, because on every E-13B font checked so far these two are swapped
# relative to the actual glyph shapes.
UNICODE_OCR_CODEPOINTS = {
    "transit": 0x2446,  # OCR BRANCH BANK IDENTIFICATION
    "amount": 0x2447,  # OCR AMOUNT OF CHECK
    "dash": 0x2448,  # OCR DASH        -- actually draws the on-us symbol
    "onus": 0x2449,  # OCR CUSTOMER ACCOUNT NUMBER -- actually draws the dash
}

# Most commercial/free E-13B TTFs do not map the OCR block at all and instead
# put the four symbols on spare ASCII keys. These are the common layouts.
ASCII_CANDIDATES: tuple[tuple[str, dict[str, str]], ...] = (
    ("ascii-abcd-lower", {"transit": "a", "amount": "b", "onus": "c", "dash": "d"}),
    ("ascii-ABCD-upper", {"transit": "A", "amount": "B", "onus": "C", "dash": "D"}),
    ("ascii-dcba", {"transit": "d", "amount": "c", "onus": "b", "dash": "a"}),
    ("ascii-taod", {"transit": "t", "amount": "a", "onus": "o", "dash": "d"}),
    ("ascii-TAOD", {"transit": "T", "amount": "A", "onus": "O", "dash": "D"}),
    ("ascii-punct", {"transit": ";", "amount": "'", "onus": "!", "dash": "-"}),
)


def codepoints_to_chars(codepoints: dict[str, int]) -> dict[str, str]:
    return {name: chr(cp) for name, cp in codepoints.items()}


def full_charmap(symbol_chars: dict[str, str]) -> dict[str, str]:
    """Expand a symbol-only mapping into all 14 classes (digits map to ASCII)."""
    charmap = {d: d for d in DIGIT_CLASSES}
    charmap.update(symbol_chars)
    missing = [c for c in CLASSES if c not in charmap]
    if missing:
        raise ValueError(f"charmap is missing classes: {missing}")
    return charmap


def micr_line_to_labels(line: str) -> list[str]:
    """Turn a labels.txt MICR string into a list of class names.

    Accepts either the substitution scheme (``O001234O T123456780T ...``) or
    the raw unicode glyphs. Spaces are field separators and are dropped.
    """
    reverse_sub = {v: k for k, v in SUBSTITUTION.items()}
    glyph_to_class = {chr(cp): name for name, cp in SPEC_CODEPOINTS.items()}
    glyph_to_class.update({chr(cp): name for name, cp in UNICODE_OCR_CODEPOINTS.items()})

    labels: list[str] = []
    for ch in line.strip():
        if ch.isspace():
            continue
        if ch in glyph_to_class:
            labels.append(glyph_to_class[ch])
        elif ch in reverse_sub:
            labels.append(reverse_sub[ch])
        else:
            raise ValueError(f"unrecognised MICR character {ch!r} in line: {line!r}")
    return labels
