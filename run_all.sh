#!/usr/bin/env bash
# One command: clone -> trained model -> pushed back to GitHub.
#
#   bash run_all.sh                 # build, train, export
#   bash run_all.sh --push          # ... and push the model back to the repo
#
# Written for a headless Ubuntu box where you cannot copy files off. With
# --push the finished .tflite, labels and training report are committed and
# pushed, so you collect them from GitHub instead.
#
# Pushing needs a token, since there is no browser to authenticate with:
#
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxx
#   bash run_all.sh --push
#
# Create one at https://github.com/settings/tokens with 'repo' scope. It is
# used for this push only and never written to disk.
#
# Everything the model needs is in the repo already -- the E-13B font and the
# synthetic training set. Real check photos are NOT required to train: the
# model learns from rendered glyphs. They only add a test_real accuracy number,
# and they are deliberately not in this repo because they carry customer names,
# addresses and account numbers.

set -euo pipefail

PUSH=0
EPOCHS=20
VERSION="v1"
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=1 ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --version=*) VERSION="${arg#*=}" ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

cd "$(dirname "$0")"
START=$(date +%s)

# --- 1. environment ---------------------------------------------------------
if [ ! -d .venv ]; then
    echo "==> no .venv, running setup_ubuntu.sh"
    bash setup_ubuntu.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate

WORKERS=$(nproc 2>/dev/null || echo 4)
echo "==> $(python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())')"
echo "==> $WORKERS cores"

# --- 2. dataset -------------------------------------------------------------
# If the synthetic set came down with the clone, build reuses it and this is a
# no-op. If not, it renders in about two minutes.
echo
echo "==> building dataset"
python -m micr.build --workers "$WORKERS"

# --- 3. train + export ------------------------------------------------------
REAL_ARGS=()
if compgen -G "dataset/train_real/*/*.png" >/dev/null 2>&1; then
    echo "==> real crops found, mixing them into training"
    REAL_ARGS=(--real-data dataset/train_real --real-weight 0.25)
fi

echo
echo "==> training ($EPOCHS epochs)"
python -m micr.train --data dataset --epochs "$EPOCHS" --version "$VERSION" \
    --workers "$WORKERS" "${REAL_ARGS[@]}"

echo
echo "==> exporting"
python -m micr.export --checkpoint "models/micr_cnn_${VERSION}.pth" \
    --version "$VERSION" --calib-data dataset/val --int8

if compgen -G "dataset/test_real/*/*.png" >/dev/null 2>&1; then
    echo
    echo "==> evaluating on held-out real checks"
    python -m micr.evaluate --checkpoint "models/micr_cnn_${VERSION}.pth" \
        --data dataset/test_real --json "models/micr_cnn_${VERSION}_test_real.json" || true
fi

ELAPSED=$(( ($(date +%s) - START) / 60 ))
echo
echo "=================================================================="
echo "done in ${ELAPSED} min"
ls -lh "export/micr_cnn_${VERSION}.tflite" export/micr_labels.json 2>/dev/null || true

# --- 4. push the model back -------------------------------------------------
if [ "$PUSH" != "1" ]; then
    echo
    echo "Model is in export/. Re-run with --push to send it back to GitHub."
    exit 0
fi

echo
echo "==> pushing model artefacts"
git config user.name  "$(git config user.name  || echo MelodyStack)" >/dev/null 2>&1 || true
git config user.email "$(git config user.email || echo mju34170@gmail.com)" >/dev/null 2>&1 || true

git add -f "models/micr_cnn_${VERSION}.pth" \
           "models/micr_cnn_${VERSION}_training_report.json" \
           "export/micr_cnn_${VERSION}"*.tflite \
           "export/micr_cnn_${VERSION}.onnx" \
           export/micr_labels.json export/micr_labels.txt 2>/dev/null || true
[ -f "models/micr_cnn_${VERSION}_test_real.json" ] && \
    git add -f "models/micr_cnn_${VERSION}_test_real.json"

ACC=$(python - <<PY
import json, pathlib
p = pathlib.Path("models/micr_cnn_${VERSION}_training_report.json")
print(f"{json.loads(p.read_text())['best_val_accuracy']*100:.3f}%" if p.is_file() else "n/a")
PY
)

if git diff --cached --quiet; then
    echo "  nothing changed, not committing"
else
    git commit -q -m "Trained model ${VERSION} (val ${ACC})

Produced by run_all.sh on the training box. Includes the PyTorch checkpoint,
the ONNX intermediate, float32/float16/int8 TFLite builds, the class labels
and the training report."
    echo "  committed"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ -n "${GITHUB_TOKEN:-}" ]; then
    URL=$(git remote get-url origin | sed -E "s#https://(.*@)?#https://${GITHUB_TOKEN}@#")
    git push "$URL" "$BRANCH" 2>&1 | sed "s/${GITHUB_TOKEN}/***/g"
else
    echo "  GITHUB_TOKEN not set, trying the stored credential helper"
    git push origin "$BRANCH"
fi

echo
echo "Collect the model from:"
echo "  https://github.com/MelodyStack/micr-training/tree/${BRANCH}/export"
