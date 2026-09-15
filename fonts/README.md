# E-13B fonts

`python -m micr.build` picks up every `.ttf` / `.otf` in this folder. Drop more
than one in and the synthetic set is rendered across all of them, which helps —
real checks are not all printed with the same face.

## What is here

**`E13B-OFL.ttf`** — built from the SVG outlines at
[zaxbux/MICR_E13-B_Font](https://github.com/zaxbux/MICR_E13-B_Font), which were
drawn from Payments Canada Standard 006 / ISO 1004:1995.

* Licence: **SIL Open Font License 1.1** (`E13B-OFL.LICENSE.md`), © 2021
  Zachary Schneider. Bundling, embedding, modification and sale *with software*
  are all permitted. No Reserved Font Name is declared.
* The upstream repo ships SVGs only. `tools/build_e13b_font.py` compiles them
  into this TTF with fontTools — no FontForge needed. Sources are kept in
  `E13B-OFL.svgs/` so the build is reproducible:

  ```
  python tools/build_e13b_font.py --svg-dir fonts/E13B-OFL.svgs \
      --out fonts/E13B-OFL.ttf --align center
  ```

* Metrics follow ISO 1004: 0.125 in character pitch, 0.117 in character height.
  One em is one character pitch, so the font is monospaced like real E-13B.
  `--align center` matters — see below.

### Verified against real print

Every glyph was compared against the MICR band of a real check (`checks/chk001.jpg`):

* `transit` and `onus` shapes match the real symbols exactly.
* The on-us symbol measures 0.78× digit height on the real check; this font
  gives 0.775×.
* Short glyphs are **vertically centred**, not sat on the baseline. Measured
  from the real check, on-us centres at 0.49–0.50 of band height; bottom-aligned
  it rendered at 0.59. Hence `--align center`.

A model trained on this font's synthetic output alone — never shown a real
crop — read all 32 glyphs of `chk001` correctly.

### The codepoint trap

Unicode's names for the OCR block are misleading. It calls U+2448 "OCR DASH"
and U+2449 "OCR CUSTOMER ACCOUNT NUMBER", but the glyphs are the other way
round. The verified mapping, which `micr/classes.py` now uses:

| class | codepoint | Unicode's name for it |
|---|---|---|
| transit | U+2446 | OCR BRANCH BANK IDENTIFICATION ✓ |
| amount | U+2447 | OCR AMOUNT OF CHECK ✓ |
| **onus** | **U+2448** | "OCR DASH" ✗ |
| **dash** | **U+2449** | "OCR CUSTOMER ACCOUNT NUMBER" ✗ |

Trusting the Unicode names produces a model that reports "dash" for every on-us
delimiter — on every check. Spec section 4 has U+2448 as on-us and is right;
only its dash codepoint (U+2444) is off.

## Adding a commercial font

If the client licenses one (spec section 11 budgets $20–$200), drop it in here
and rebuild. Do **not** commit it unless its licence allows redistribution —
`.gitignore` excludes everything in this folder except the OFL build.

Fonts disagree about where the four symbols live: some use the OCR block, most
put them on spare ASCII keys. The resolver probes the cmap and tries known
layouts in order, verified mapping first. If it picks wrong, override it:

```json
{ "transit": "a", "amount": "b", "onus": "c", "dash": "d" }
```

```
python -m micr.fonts --font fonts/NewFont.ttf --preview check.png
python -m micr.build --charmap charmap.json
```

**Always open `dataset/charmap_preview.png` after changing fonts.** It is the
one step that cannot be automated, and it is exactly what caught the
onus/dash swap above.
