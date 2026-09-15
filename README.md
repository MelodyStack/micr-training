# MICR E-13B model: training and export

Implements sections 4–7 of `micr-scanner-spec.md`: render a synthetic E-13B
dataset, train the 14-class CNN, export it to `.tflite` for Android (and
optionally `.mlpackage` for iOS later).

Python is used **only** for training and conversion. Nothing here ships with
the React Native app — the app gets two files: `micr_cnn_v1.tflite` and
`micr_labels.json`.

## On a headless Ubuntu box: one command, model pushed back

```bash
git clone https://github.com/MelodyStack/micr-training.git
cd micr-training
bash run_all.sh --push
```

Installs everything, builds the dataset, trains, exports `.tflite`, then commits
the model back to this repo so you can collect it from GitHub rather than
copying files off the server. Roughly 30–60 minutes on a CPU instance.

Pushing needs a token, since a headless box has no browser to authenticate
with. Create one at <https://github.com/settings/tokens> with `repo` scope:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
bash run_all.sh --push
```

It is used for that push only and never written to disk. Without `--push` the
model is simply left in `export/`.

The model artefacts land in [`export/`](export/): `micr_cnn_v1.tflite` (the
Android file), the fp16 and int8 builds, and `micr_labels.json`.

**Real check photos are not needed to train.** The model learns from glyphs
rendered from the E-13B font, both of which are in this repo. A model trained
on synthetic data alone, having never seen a real crop, read all 32 glyphs of a
real check correctly. Real checks only add a `test_real` accuracy number — and
they are deliberately absent here, because they carry customer names,
addresses, signatures and account numbers. If you have them locally, drop them
in `checks/` and `run_all.sh` picks them up automatically.

## The whole thing in one command

You add files to two folders. Everything else is generated.

```
fonts/E13B-OFL.ttf          the E-13B font             <- already here
checks/chk001.jpg           check photos               <- you
checks/labels.txt           one MICR line per photo    <- you
```

```powershell
python -m micr.build
```

```
dataset/train/              synthetic, balanced        <- generated
dataset/val/                synthetic, balanced        <- generated
dataset/train_real/         crops from your checks     <- generated
dataset/test_real/          crops from held-out checks <- generated
dataset/charmap_preview.png the 14 glyphs, to eyeball  <- generated
debug/chk001_debug.png      band with boxes + labels   <- generated
```

Drop in more checks and run it again. The synthetic set is only re-rendered when
a setting that produced it changes, so a normal re-run takes seconds instead of
re-rendering 56,000 images; it tells you which setting changed when it does
re-render. Check crops are always rebuilt, because that is cheap and keeps them
consistent with whatever `labels.txt` says right now.

`--train` continues into training, `--export` goes on to the `.tflite`:

```powershell
python -m micr.build --train --export
```

The sections below are the same pipeline run stage by stage, which is what you
want when something needs adjusting.

## What is and is not in here

| Spec section | Status |
|---|---|
| 4 — the 14 classes | `micr/classes.py`, single source of truth |
| 5 — training tooling | this package |
| 6 — export filenames | `micr/export.py` |
| 7 — synthetic data, `ImageFolder` layout | `micr/synth.py`, `micr/dataset.py` |
| 9 — segmentation of real checks | `micr/segment.py` |
| 10 — runtime pipeline | **not included** — React Native / native side |

`micr/build.py` drives all of it; the rest stay usable on their own.

## Install — Ubuntu / EC2

```bash
bash setup_ubuntu.sh
source .venv/bin/activate
```

Picks the CUDA or CPU torch build based on whether the instance has a GPU, then
verifies the install. Three things that differ from the Windows setup and will
bite otherwise:

* **`opencv-python-headless`, not `opencv-python`.** The GUI build links
  `libGL.so.1`, which server images do not ship. The failure is an `ImportError`
  the first time anything touches cv2 — i.e. several minutes into data
  generation.
* **`--workers 8` on training.** The Windows default is 0 because spawning
  DataLoader workers there is slow; on Linux fork makes them cheap and you
  should use them. Data generation already uses every core.
* **Upload the E-13B font.** `scp` it to `fonts/` and check the contact sheet
  before generating anything — there is no system font to fall back on.

OpenCV's internal thread pool is disabled inside worker processes, so a large
instance will not oversubscribe itself.

On instance choice: spec section 5 notes AWS/GCP GPU instances run several times
the price of RunPod or Vast for the same card. For a 14-class problem on 48×32
crops a plain CPU instance is genuinely enough — 20 epochs is well under an hour
on a modest box, and `--device auto` picks up a GPU if one is there.

## Install — Windows

`.venv\` in this directory is already set up with everything, including the
TFLite toolchain — activate it and skip ahead:

```powershell
cd d:\Work\mustafa\micr-training
.\.venv\Scripts\Activate.ps1
```

To rebuild it from scratch:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tensorflow tf_keras onnx2tf onnx-graphsurgeon sng4onnx
```

`tf_keras` is not optional despite the name: onnx2tf needs it to emit the
saved_model that the fp16 and int8 conversions are built from. Without it you
still get a working float32 `.tflite`, just not the smaller variants.

If TensorFlow will not install on some other machine, run
`python -m micr.export --onnx-only` there and do the ONNX → TFLite step on WSL,
Linux or Colab. The `.onnx` file is portable; nothing else has to move.

> **Windows runtime note.** PyTorch failed to load on this machine until the
> Microsoft Visual C++ redistributable was repaired — something had overwritten
> `C:\Windows\System32\msvcp140.dll` with a 2015-era build while every sibling
> DLL was current, so `c10.dll` could not initialise. Fixed by installing
> [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) (now
> 14.44). If `import torch` ever dies with `WinError 1114`, check that file's
> version first.

## The E-13B font

**Already in `fonts/E13B-OFL.ttf`.** Built from
[zaxbux/MICR_E13-B_Font](https://github.com/zaxbux/MICR_E13-B_Font) — outlines
drawn from Payments Canada Standard 006 / ISO 1004:1995, **SIL OFL 1.1**, which
permits bundling, embedding and sale with software. The upstream repo ships SVGs
only; `tools/build_e13b_font.py` compiles them to a TTF with fontTools. Sources
and licence are kept alongside it. See [fonts/README.md](fonts/README.md).

This removes the $20–$200 line item in spec section 11, and it is verified
rather than assumed: every glyph was compared against the MICR band of
`checks/chk001.jpg`, and a model trained on its synthetic output alone read all
32 glyphs of that check correctly. Swapping in a commercial font later is one
file drop plus a rebuild.

If you do add another font, **verify the glyph mapping before generating
60,000 images**:

```powershell
python -m micr.fonts --font fonts\E13B.ttf --preview charmap_preview.png
```

This prints which character each of the 14 classes maps to and writes a
labelled contact sheet. Open it. Every glyph must look like real E-13B print.

Fonts disagree about where the four symbols live: some use the Unicode OCR
block, most put them on spare ASCII keys (`a b c d`, `A B C D`, …). The
resolver probes the font's cmap and tries the known layouts in order. If it
picks the wrong one, or finds none, override it:

```json
// charmap.json
{ "transit": "a", "amount": "b", "onus": "c", "dash": "d" }
```

```powershell
python -m micr.fonts --font fonts\E13B.ttf --charmap charmap.json --preview check.png
```

Pass the same `--charmap` to `micr.synth`.

> **The codepoint trap — settled.** Unicode names `U+2448` "OCR DASH" and
> `U+2449` "OCR CUSTOMER ACCOUNT NUMBER", but compared against the real glyphs
> on `chk001` those two are **swapped**: `U+2448` draws the on-us symbol and
> `U+2449` draws the dash. Spec section 4 is right that on-us is `U+2448`; only
> its dash codepoint (`U+2444`, actually OCR BELT BUCKLE) is wrong.
>
> `micr/classes.py` now leads with the verified mapping. This is not academic:
> the resolver originally picked the Unicode-named map, which would have
> produced a model reporting "dash" for every on-us delimiter — on every check.
> The contact sheet is what caught it, which is why the instruction to open it
> is not a formality.

## 1. Generate the dataset

```powershell
python -m micr.synth --font fonts\E13B.ttf --out dataset `
    --train-per-class 4000 --val-per-class 500
```

56,000 train + 7,000 val crops, roughly 3–6 minutes on a multi-core CPU. Each
sample is one glyph rendered at 4× then area-downscaled to 32×48 grayscale,
composited onto randomised paper (illumination gradient, grain, tint lines)
with randomised ink weight, then run through the heavy augmentation pipeline:
blur, skew, shadow, elastic warp, JPEG artefacts, ink dropout. Roughly a
quarter of crops get slivers of the neighbouring glyph bleeding in at the
edges, because the runtime segmenter will not cut perfectly either.

Output is exactly the layout the spec calls for — folder name *is* the label:

```
dataset/
  train/0/ 1/ ... 9/ amount/ dash/ onus/ transit/
  val/  0/ 1/ ... 9/ amount/ dash/ onus/ transit/
  test_real/  (empty skeleton + labels.txt template)
  charmap_preview.png
  dataset_meta.json
```

Useful flags: `--font` repeatedly to mix several E-13B faces, `--aug-strength`,
`--neighbor-bleed-prob`, `--no-augment`, `--workers`.

## 2. Train

```powershell
python -m micr.train --data dataset --epochs 20 --version v1
```

→ `models/micr_cnn_v1.pth` (best val epoch) and a JSON training report.

The model is ~123k parameters, ~490 KB fp32. Six conv layers, three pools, a
128-unit head. Input normalisation is **inside** the model, so the app hands
over pixels divided by 255 and nothing else. AdamW + OneCycle + label
smoothing, early stopping on val accuracy. CUDA and AMP are used if present;
CPU-only is fine and takes roughly 20–40 minutes for 20 epochs.

Per-epoch light augmentation runs on top of the baked-in augmentation so the
model does not memorise the frozen variants (`--online-aug eval` disables it).

## 3. Add real check photos

Photos go in `checks/`, one line per photo in `checks/labels.txt`. That is the
whole manual step — the folder structure underneath is generated.

**Bringing in a batch:** the check id is the filename stem, and `labels.txt` is
whitespace-separated, so a filename with spaces in it (`WhatsApp Image 2026-09-14
at 10.16.01 PM.jpeg`) truncates the id at the first space and matches nothing.
Every such check would report "NO LABEL" and contribute no crops. Use `ingest`
rather than renaming by hand:

```powershell
python -m micr.ingest --from "D:\photos\client-checks"
```

```
  chk002  <- IMG_20260914_223344.jpg
  chk003  <- WhatsApp Image 2026-09-11 at 10.16.01 PM.jpeg
  ...
100 added, 0 skipped -> checks
```

Copies (never moves, unless `--move`), numbers from the first free id, and never
overwrites an existing photo. `checks/sources.csv` records the mapping back to
the original filename so a bad crop is always traceable. Re-running is safe:
already-ingested files are recognised by content hash and skipped, so you can
re-scan the same folder after more photos arrive. Commented stubs are appended
to `labels.txt`, one per new check, ready to fill in.

```
checks/
  chk001.jpg
  chk002.jpg
  labels.txt        <- chk001  O001234O T123456780T 000123456789O
```

The check id is the filename without its extension. `T`/`A`/`O`/`D` are the
transit, amount, on-us and dash symbols; spaces separate fields and are ignored.

```powershell
python -m micr.segment --checks checks `
    --out dataset\train_real --holdout-out dataset\test_real --holdout-frac 0.3 `
    --debug-dir debug
```

Locate band → deskew → binarize → cut → file each crop by its label:

```
dataset/train_real/3/chk001_pos11.png
dataset/test_real/onus/chk007_pos00.png
```

**Whole checks go to one split or the other, never both.** Two glyphs cut from
the same photo share its paper, print run, lighting and camera; letting one land
in train and the other in test leaks, and the accuracy you report comes out
flattering and wrong. The assignment is hashed from the check id, so it stays
stable as you add more checks. Drop `--holdout-out` to send everything to one
place.

### Then mix them into training

```powershell
python -m micr.train --data dataset --real-data dataset\train_real --real-weight 0.25
```

```
train:      56,000 synthetic   online aug: online
real:       2,144 crops x7 = 15,008 (21% of each epoch, target 25%)
            no real samples for ['amount'] -- synthetic only there
```

A few hundred real crops sitting next to 56,000 synthetic ones would contribute
almost nothing to the gradient, so they are repeated to reach `--real-weight` of
each epoch. Repetition rather than a sampler keeps the epoch length honest and
the augmentation fresh: every repeat draws different jitter, so the model sees
variations of the real crop rather than the same tensor seven times.

**Real crops supplement the synthetic set, they do not replace it.** One check
yields 32 glyphs with four classes at zero — train on that alone and the model
cannot learn `4`, `9`, `amount` or `dash` at all. Synthetic data provides
balanced bulk; real crops teach it what actual paper and print look like. The
class coverage of your real data is printed at startup so the gaps are visible.

### One check at a time

```powershell
python -m micr.segment --image checks\chk001.jpg --check-id chk001 `
    --micr "O001234O T123456780T 000123456789O" `
    --out dataset\test_real --debug-dir debug
```

### How it works, and two design decisions

The segmenter runs the same sequence the app must run on a camera frame —
locate band, deskew, binarize, find character boundaries — which is the point of
writing it here rather than only for training.

* **Crops are full band height**, not each glyph's own bounding box. The
  synthetic renderer puts every glyph on one shared baseline at its true
  relative height, so the dash is short *within its crop*. Cropping tight here
  would rescale the dash to full height and hand the model test data that looks
  nothing like what it trained on.
* **Glyph boundaries come from a fixed-pitch grid fit**, not from gap size.
  Gap-based merging cannot work: the transit and on-us symbols are drawn as
  several separate vertical strokes, so any threshold loose enough to join one
  symbol's strokes also joins two adjacent digits. Both failure modes showed up
  on the first real check tried. E-13B's constant pitch resolves it — strokes
  of one symbol land in one cell, adjacent characters do not, and the blank
  cells between fields simply hold no ink. Isolated marks far from the line
  (the check's printed border) are dropped.

**Nothing is written unless the glyph count matches the label exactly.** A
segmenter that finds 31 glyphs in a 32-glyph line would file every crop after
the miss under the wrong class — worse than no data at all. On a mismatch it
says so, writes the debug image, and moves on. `--dry-run` checks without
writing; `--debug-dir` draws the detected boxes with their assigned labels.

If a check misses, look at the debug image first, then try `--adaptive` (low
contrast), `--search-frac` (band outside the bottom 42%), or `--pitch-min-frac`.

Re-running after changing the segmentation code costs nothing — that is what
`labels.txt` is for. Nothing is ever re-labelled by hand.

## 4. Evaluate on real checks

```powershell
python -m micr.evaluate --checkpoint models\micr_cnn_v1.pth --data dataset\test_real
```

Val accuracy is our own renderer marking its own homework. **Quote the client
the `test_real` number** — checks the model was never trained on, which is what
`--holdout-out` in step 3 is for.

The report prints per-class accuracy, the top confusions, and the compounded
30-glyph line accuracy — which is the number that actually predicts how often a
scan passes the ABA checksum first try. 99.5% per glyph is only 86% per line.

Classes with no `test_real` samples are reported as `n/a`, not as 100%. The
spec's own worked example produced no 4s, no 9s, no amount symbol and no dash
from one check — collect business and payroll checks or the amount symbol
column stays empty.

## 5. Export

```powershell
python -m micr.export --checkpoint models\micr_cnn_v1.pth --version v1
```

Writes to `export/`:

| File | Purpose |
|---|---|
| `micr_cnn_v1.onnx` | intermediate |
| `micr_cnn_v1.tflite` | float32, ~486 KB — **the Android file** |
| `micr_cnn_v1_fp16.tflite` | ~247 KB, float32 I/O, fp16 weights |
| `micr_cnn_v1_int8.tflite` | `--int8`, ~135 KB, full-integer with float I/O |
| `micr_labels.json` / `.txt` | class order + T/A/O/D substitution |
| `MicrCNN_v1.mlpackage` | `--coreml`, iOS later |
| `_onnx2tf_v1/` | conversion scratch, safe to delete |

Every produced file is checked numerically against PyTorch on real crops and
the argmax agreement is printed. This matters: a silently transposed or badly
quantised model still returns plausible-looking digits, and the ABA checksum
will reject every read with no indication why. fp32/fp16 should be 100%; int8
around 99.7% is normal and is not flagged.

The fp16 and int8 files are deliberately **not** onnx2tf's own `_float16`
output. That one makes the input tensor float16, which the TFLite CPU CONV_2D
kernel refuses outright — it only runs under a GPU delegate. These are
converted from the saved_model instead, so they keep float32 I/O and run
anywhere.

Other flags: `--onnx-only`, `--no-fp16`, `--opset`, `--calib-data`.

## Model contract for the app

```
input   float32  [1, 48, 32, 1]  NHWC, grayscale, 0.0–1.0
output  float32  [1, 14]         raw logits — softmax for a confidence score
```

Class index order (also in `micr_labels.txt`):

```
0  1  2  3  4  5  6  7  8  9  amount  dash  onus  transit
```

Substitution for the assembled string: `transit`→`T`, `amount`→`A`,
`onus`→`O`, `dash`→`D`.

Core ML uses `[1, 1, 48, 32]` NCHW instead — the only difference between the
two platforms.

## What has been verified

Whole pipeline run end to end on a stand-in font (Arial, whose `a b c d` the
resolver picks up as the four symbols) — generate → train → evaluate → export:

* 123,102 parameters, 481 KB fp32, as designed
* training converges, checkpoints, early-stops, writes its report
* `test_real` with empty **and** entirely missing class folders evaluates
  correctly, with class indices staying aligned
* ONNX 100% argmax parity with PyTorch, max logit delta 1.9e-06
* TFLite fp32 and fp16 100% parity, int8 99.67%, all `[1,48,32,1]` NHWC float32

Not yet run against a real E-13B font or real check crops — that needs the
licensed font and the client's sample checks.

## Retraining

Version the filenames (`--version v2`). Retraining after real photos expose
failure cases is expected, and it has to stay obvious which build shipped which
weights. `classes.py` is the contract: if the class list or its order ever
changes, every previously exported `.tflite` becomes wrong, and `evaluate.py`
and `export.py` both refuse to load a checkpoint whose order disagrees.
