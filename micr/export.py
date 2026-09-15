"""Export a trained checkpoint to the mobile runtime formats.

Path, per spec section 6:  .pth  ->  ONNX  ->  .tflite

    python -m micr.export --checkpoint models/micr_cnn_v1.pth --version v1

Produces in ``export/``:
    micr_cnn_v1.onnx          intermediate
    micr_cnn_v1.tflite        float32, Android  <- the shipping file
    micr_cnn_v1_fp16.tflite   half the size, usually identical accuracy
    micr_cnn_v1_int8.tflite   optional, --int8 (needs calibration images)
    micr_labels.json          class order + T/A/O/D substitution for the app
    micr_labels.txt           one class per line, index order
    MicrCNN_v1.mlpackage      optional, --coreml (iOS, later)

Every stage is verified numerically against PyTorch, because a silently
transposed or wrongly-quantised model still returns plausible digits -- and
the ABA checksum will just reject every read with no clue why.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch

from .classes import (
    CLASSES,
    INPUT_HEIGHT,
    INPUT_MEAN,
    INPUT_STD,
    INPUT_WIDTH,
    SUBSTITUTION,
)
from .evaluate import load_checkpoint
from .model import example_input

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


# --- calibration / verification data ---------------------------------------


def load_sample_images(data_dir: Path, limit: int) -> np.ndarray:
    """Load up to ``limit`` crops as (N, 1, H, W) float32 in [0, 1], NCHW."""
    import cv2

    paths = sorted(data_dir.rglob("*.png")) + sorted(data_dir.rglob("*.jpg"))
    if not paths:
        raise SystemExit(f"no images found under {data_dir}")

    # Spread across classes rather than taking the first N of class "0".
    step = max(1, len(paths) // limit)
    chosen = paths[::step][:limit]

    batch = np.empty((len(chosen), 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
    for i, path in enumerate(chosen):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {path}")
        if img.shape != (INPUT_HEIGHT, INPUT_WIDTH):
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        batch[i, 0] = img.astype(np.float32) / 255.0
    return batch


# --- stage 1: ONNX ---------------------------------------------------------


def export_onnx(model: torch.nn.Module, out_path: Path, opset: int, simplify: bool) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = example_input(1)

    kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset,
        do_constant_folding=True,
    )
    with warnings.catch_warnings():
        # torch >= 2.9 warns that the dynamo exporter is now the default. The
        # legacy tracer is a deliberate choice here: it emits the flat conv/BN
        # graph onnx2tf converts most predictably. Silence the nag, keep the note.
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
        try:
            torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(out_path), **kwargs)

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            simplified, ok = onnx_simplify(onnx.load(str(out_path)))
            if ok:
                onnx.save(simplified, str(out_path))
                print("  onnxsim: simplified")
            else:
                print("  onnxsim: check failed, keeping the unsimplified graph")
        except ImportError:
            print("  onnxsim not installed, skipping simplification")

    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(out_path)))
        print("  onnx.checker: ok")
    except ImportError:
        pass
    return out_path


def verify_onnx(onnx_path: Path, model: torch.nn.Module, samples: np.ndarray) -> float:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping ONNX parity check")
        return float("nan")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(samples)).numpy()

    onnx_logits = np.concatenate(
        [session.run(None, {input_name: samples[i : i + 1]})[0] for i in range(len(samples))]
    )
    agreement = float(
        (onnx_logits.argmax(axis=1) == torch_logits.argmax(axis=1)).mean()
    )
    max_delta = float(np.abs(onnx_logits - torch_logits).max())
    print(
        f"  ONNX vs PyTorch: {agreement * 100:.2f}% argmax agreement, "
        f"max logit delta {max_delta:.2e}"
    )
    return agreement


# --- stage 2: TFLite -------------------------------------------------------


def onnx_to_tf(onnx_path: Path, work_dir: Path) -> tuple[Path | None, dict[str, Path]]:
    """Run onnx2tf; return (saved_model dir or None, {precision: tflite file}).

    onnx2tf changed shape here: up to 2.5 it emitted a saved_model that you then
    converted yourself, from 2.6 it writes the float32/float16 .tflite straight
    out of the graph and skips the saved_model unless asked. Both are handled --
    its own outputs are used when present, and the saved_model is still
    requested because int8 calibration needs one.
    """
    try:
        import onnx2tf
    except ImportError as exc:
        raise SystemExit(
            "onnx2tf is not installed (it pulls in tensorflow).\n"
            "  pip install onnx2tf tensorflow onnx onnx-graphsurgeon sng4onnx\n"
            "If tensorflow will not install on this machine, the ONNX file above is\n"
            "portable: run this same command on WSL, Linux or Colab to get the .tflite."
        ) from exc

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "input_onnx_file_path": str(onnx_path),
        "output_folder_path": str(work_dir),
    }
    optional = {
        "copy_onnx_input_output_names_to_tflite": True,
        "output_signaturedefs": True,
        "non_verbose": True,
        "flatbuffer_direct_output_saved_model": True,
    }

    # onnx2tf wraps convert() as (*args, **kwargs), so signature introspection
    # reports no named parameters at all -- filtering against it would silently
    # drop every argument. Only filter when the signature is actually concrete.
    params = inspect.signature(onnx2tf.convert).parameters
    variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if variadic:
        kwargs = {**required, **optional}
    else:
        kwargs = {k: v for k, v in {**required, **optional}.items() if k in params}

    try:
        onnx2tf.convert(**kwargs)
    except Exception as exc:
        # An optional flag this build cannot honour -- an unknown keyword
        # (TypeError) or a missing extra, e.g. saved_model export needs
        # tf_keras. Fall back to a plain conversion; a genuine graph failure
        # will surface again from the retry.
        print(f"  onnx2tf rejected the optional flags ({type(exc).__name__}: {exc})")
        print("  retrying with defaults")
        onnx2tf.convert(**required)

    saved_model = next((p.parent for p in work_dir.rglob("saved_model.pb")), None)

    produced: dict[str, Path] = {}
    for precision, suffix in (("fp32", "_float32.tflite"), ("fp16", "_float16.tflite")):
        match = next(iter(sorted(work_dir.rglob(f"*{suffix}"))), None)
        if match is not None:
            produced[precision] = match

    if saved_model is None and not produced:
        raise SystemExit(
            f"onnx2tf produced neither a saved_model nor a .tflite under {work_dir}"
        )
    return saved_model, produced


def convert_tflite(
    saved_model_dir: Path,
    out_path: Path,
    precision: str,
    representative: np.ndarray | None = None,
) -> Path:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if precision == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif precision == "int8":
        if representative is None:
            raise ValueError("int8 needs a representative dataset")
        nhwc = np.transpose(representative, (0, 2, 3, 1)).astype(np.float32)

        def representative_dataset():
            for i in range(len(nhwc)):
                yield [nhwc[i : i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Float IO keeps the React Native side simple: no scale/zero-point
        # arithmetic in JS, only the weights and activations are int8.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(converter.convert())
    return out_path


def verify_tflite(tflite_path: Path, model: torch.nn.Module, samples: np.ndarray) -> float:
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    except ImportError:
        print(f"  tensorflow missing, cannot verify {tflite_path.name}")
        return float("nan")

    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_shape = [int(d) for d in in_detail["shape"]]

    with torch.no_grad():
        torch_pred = model(torch.from_numpy(samples)).numpy().argmax(axis=1)

    preds = np.empty(len(samples), dtype=np.int64)
    for i in range(len(samples)):
        x = samples[i : i + 1]  # NCHW
        if in_shape[-1] == 1 and in_shape[1] == INPUT_HEIGHT:
            x = np.transpose(x, (0, 2, 3, 1))  # -> NHWC
        interpreter.set_tensor(in_detail["index"], x.astype(in_detail["dtype"]))
        interpreter.invoke()
        preds[i] = int(np.argmax(interpreter.get_tensor(out_detail["index"])[0]))

    agreement = float((preds == torch_pred).mean())
    layout = "NHWC" if in_shape[-1] == 1 else "NCHW"
    print(
        f"  {tflite_path.name}: input {in_shape} {layout} {in_detail['dtype'].__name__}, "
        f"{agreement * 100:.2f}% argmax agreement with PyTorch"
    )
    return agreement


# --- stage 3: labels + Core ML ---------------------------------------------


def write_labels(out_dir: Path, version: str, checkpoint: dict) -> None:
    labels_json = out_dir / "micr_labels.json"
    labels_json.write_text(
        json.dumps(
            {
                "version": version,
                "classes": list(CLASSES),
                "substitution": {c: SUBSTITUTION[c] for c in CLASSES},
                "input": {
                    "height": INPUT_HEIGHT,
                    "width": INPUT_WIDTH,
                    "channels": 1,
                    "layout_tflite": "NHWC",
                    "layout_coreml": "NCHW",
                    "dtype": "float32",
                    "range": [0.0, 1.0],
                    "note": (
                        f"Normalisation ((x - {INPUT_MEAN}) / {INPUT_STD}) is baked into "
                        "the model. Feed grayscale pixels divided by 255 and nothing else."
                    ),
                },
                "output": {
                    "name": OUTPUT_NAME,
                    "shape": [1, len(CLASSES)],
                    "note": "raw logits, apply softmax for a confidence score",
                },
                "val_accuracy": checkpoint.get("val_accuracy"),
                "trained_at": checkpoint.get("trained_at"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "micr_labels.txt").write_text(
        "\n".join(CLASSES) + "\n", encoding="utf-8"
    )
    print(f"  labels: {labels_json.name}, micr_labels.txt")


def export_coreml(model: torch.nn.Module, out_path: Path) -> Path | None:
    try:
        import coremltools as ct
    except ImportError:
        print("  coremltools not installed, skipping Core ML (iOS comes later anyway)")
        return None

    traced = torch.jit.trace(model, example_input(1))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 1, INPUT_HEIGHT, INPUT_WIDTH))],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    mlmodel.short_description = "MICR E-13B glyph classifier"
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"  Core ML: {out_path.name}")
    return out_path


# --- driver ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the MICR model for mobile.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="export")
    parser.add_argument("--version", help="filename version tag; default: from checkpoint")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--int8", action="store_true", help="full-integer quantisation")
    parser.add_argument("--coreml", action="store_true", help="also emit .mlpackage")
    parser.add_argument("--onnx-only", action="store_true", help="stop before TFLite")
    parser.add_argument(
        "--calib-data",
        default="dataset/val",
        help="crops used for int8 calibration and for the parity checks",
    )
    parser.add_argument("--calib-samples", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cpu")  # export always traces on CPU
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    version = args.version or checkpoint.get("version", "v1")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint:  {args.checkpoint}")
    print(f"version tag: {version}")
    print(f"val acc:     {(checkpoint.get('val_accuracy') or 0) * 100:.3f}%")

    calib_dir = Path(args.calib_data)
    samples = None
    if calib_dir.is_dir():
        samples = load_sample_images(calib_dir, args.calib_samples)
        print(f"calibration: {len(samples)} crops from {calib_dir}")
    else:
        print(f"calibration: {calib_dir} not found -- using random noise for parity checks")
        samples = np.random.rand(64, 1, INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)
        if args.int8:
            raise SystemExit("--int8 needs real crops; pass --calib-data <dir>")

    print("\n[1/4] ONNX")
    onnx_path = export_onnx(
        model, out_dir / f"micr_cnn_{version}.onnx", args.opset, not args.no_simplify
    )
    verify_onnx(onnx_path, model, samples)
    print(f"  {onnx_path}  ({onnx_path.stat().st_size / 1024:.0f} KB)")

    results: list[tuple[Path, str, float]] = []
    if not args.onnx_only:
        print("\n[2/4] TFLite")
        saved_model, produced = onnx_to_tf(onnx_path, out_dir / f"_onnx2tf_{version}")
        print(f"  onnx2tf wrote: {sorted(produced) or 'saved_model only'}")

        targets = [("fp32", out_dir / f"micr_cnn_{version}.tflite")]
        if args.fp16:
            targets.append(("fp16", out_dir / f"micr_cnn_{version}_fp16.tflite"))
        if args.int8:
            targets.append(("int8", out_dir / f"micr_cnn_{version}_int8.tflite"))

        for precision, path in targets:
            if precision == "fp32" and precision in produced:
                # Already exactly what we want; no reason to convert twice.
                path.write_bytes(produced[precision].read_bytes())
            elif saved_model is not None:
                # Deliberately NOT onnx2tf's own _float16.tflite: that one makes
                # the input tensor float16, which the TFLite CPU CONV_2D kernel
                # rejects outright (it needs float32/uint8/int8/int16), so it
                # only runs under a GPU delegate. Converting from the
                # saved_model gives float16 weights with float32 I/O, which
                # runs everywhere.
                convert_tflite(saved_model, path, precision, samples)
            elif precision in produced:
                print(
                    f"  {precision}: no saved_model, falling back to onnx2tf's file -- "
                    "it has float16 I/O and needs a GPU delegate on device"
                )
                path.write_bytes(produced[precision].read_bytes())
            else:
                print(
                    f"  {precision}: skipped -- onnx2tf emitted no saved_model. "
                    "Install tf_keras, or re-run onnx2tf with "
                    "--output_integer_quantized_tflite."
                )
                continue
            agreement = verify_tflite(path, model, samples)
            results.append((path, precision, agreement))
            print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")

    print("\n[3/4] labels")
    write_labels(out_dir, version, checkpoint)

    print("\n[4/4] Core ML")
    if args.coreml:
        export_coreml(model, out_dir / f"MicrCNN_{version}.mlpackage")
    else:
        print("  skipped (--coreml to enable)")

    # Losing a crop or two is expected of int8 and means something is wrong in
    # a float build, so the bar differs by precision.
    thresholds = {"fp32": 0.999, "fp16": 0.999, "int8": 0.99}
    suspect = False
    print("\ndone.")
    for path, precision, agreement in results:
        bad = not np.isnan(agreement) and agreement < thresholds[precision]
        suspect = suspect or bad
        print(
            f"  {path.name:<32} {agreement * 100:6.2f}% parity"
            f"{'   <-- CHECK THIS' if bad else ''}"
        )
    if suspect:
        print(
            "\nParity below what this precision should reach. Re-run evaluate.py "
            "against test_real before shipping that file."
        )
    if args.onnx_only:
        print(
            f"\nStopped after ONNX: {onnx_path}\n"
            "Finish the conversion where tensorflow works (WSL, Linux, Colab):\n"
            f"  onnx2tf -i {onnx_path.name} -o tflite_out"
        )
    else:
        print(
            f"\nShip to Android: {out_dir / f'micr_cnn_{version}.tflite'} "
            f"+ {out_dir / 'micr_labels.json'}\n"
            "Load it with react-native-fast-tflite; feed one 48x32 grayscale crop per "
            "glyph, values 0..1, NHWC."
        )


if __name__ == "__main__":
    main()
