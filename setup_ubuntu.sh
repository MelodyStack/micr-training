#!/usr/bin/env bash
# Set up the MICR training environment on Ubuntu (tested target: EC2 22.04/24.04).
#
#   bash setup_ubuntu.sh
#   source .venv/bin/activate
#
# Picks the CUDA or CPU torch build based on whether the instance has a GPU,
# and installs the headless OpenCV so it does not need libGL on a server.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-dev build-essential \
    libglib2.0-0 fonts-dejavu-core

echo "==> virtualenv"
"${PYTHON_BIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU detected, installing CUDA torch"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    pip install --quiet torch torchvision
else
    echo "==> no GPU, installing CPU-only torch (fine for 14 classes on 48x32)"
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> training dependencies"
# headless, NOT opencv-python: the GUI build needs libGL.so.1, which a server
# image does not ship, and the failure is an ImportError at first use.
pip install --quiet \
    numpy pillow fonttools opencv-python-headless "albumentations>=2.0" \
    onnx onnxruntime onnxsim

echo "==> TFLite export toolchain"
# tf_keras is required for the saved_model that fp16/int8 conversion needs.
pip install --quiet tensorflow tf_keras onnx2tf onnx-graphsurgeon sng4onnx

echo
echo "==> verifying"
python - <<'PY'
import torch, torchvision, albumentations, cv2, onnx, onnxruntime
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  albumentations {albumentations.__version__}")
print(f"  opencv       {cv2.__version__}")
print(f"  onnxruntime  {onnxruntime.__version__}")
try:
    import tensorflow as tf
    print(f"  tensorflow   {tf.__version__}")
except ImportError:
    print("  tensorflow   MISSING - export will stop at ONNX")
PY

python -m micr.model

cat <<'EOF'

Done. Next:

  source .venv/bin/activate

  # 1. upload the licensed E-13B .ttf into fonts/
  # 2. upload check photos into checks/, one labels.txt line each
  # 3. build everything, then train and export:

  python -m micr.build --train --export

  # LOOK at dataset/charmap_preview.png before trusting the result -- all 14
  # glyphs must be real E-13B print, not whatever the resolver guessed.

Re-run `python -m micr.build` after adding checks; it reuses the synthetic set.
Copy export/micr_cnn_v1.tflite and export/micr_labels.json back to the app.
EOF
