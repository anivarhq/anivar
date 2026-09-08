"""FP16-quantize the installed edge models (industry-standard GPU precision).

Mature NVRs ship FP16 OpenVINO models by default and TensorRT auto-calibrates to
FP16; INT8 is reserved for Coral-class TPUs and costs measurable accuracy
(~8pp in mature NVRs' own comparisons). For Anivar's DirectML-on-GPU stack,
FP16 halves weight memory and speeds inference with negligible accuracy loss.

Writes `<name>.fp16.onnx` NEXT TO each source model — the app's session loader
prefers the sibling on GPU and falls back to FP32 automatically, so this is
opt-in and reversible (delete the .fp16.onnx to revert).

`keep_io_types=True` keeps inputs/outputs f32, so the app's tensor code is
untouched.

SKIPPED on purpose:
  • face embedder (ArcFace) — its raw-cosine thresholds are calibration-
    sensitive (recognition hard rule); revisit with a validation set.
  • CLIP text/vision — pinned to CPU, where FP16 is slower, not faster.

Usage:  python scripts/quantize_fp16.py            # converts installed skills
        python scripts/quantize_fp16.py path.onnx  # converts one model
"""
import os
import sys

try:
    import onnx
    from onnxconverter_common import float16
except ImportError:
    print("pip install onnx onnxconverter-common", file=sys.stderr)
    sys.exit(1)

SKILLS = os.path.expandvars(r"%APPDATA%\com.anivar.app\skills")

# (relative path, notes) — GPU-run models that convert CLEANLY to FP16.
#
# EXCLUDED (verified 2026-07-02): the YOLO26 detectors and the face detector —
# onnxconverter_common mangles their Cast/Resize chains ("output arg type does
# not match expected tensor(float16)" at session load), even with Resize/Upsample
# block-listed. The app's loader falls back to FP32 automatically so a bad file
# is harmless, but there's no point shipping one. Those models stay FP32 until
# they're exported at FP16 from source (e.g. ultralytics `half=True`).
TARGETS = [
    ("reid_osnet/model.onnx",    "body re-id (OSNet)"),
    ("alpr_global/model.onnx",   "licence plates"),
    ("audio_yamnet/model.onnx",  "sound classifier"),
]


def convert(src: str) -> bool:
    dst = src[: -len(".onnx")] + ".fp16.onnx"
    if os.path.exists(dst):
        print(f"  = exists  {os.path.basename(dst)}")
        return True
    try:
        model = onnx.load(src)
        # Resize/Upsample stay float32: the converter mis-types their cast chain
        # (seen on YOLO26: "output arg of Resize_output_cast0 does not match
        # expected type tensor(float16)" → the model fails to load). Keeping the
        # handful of interpolation ops in fp32 costs nothing measurable.
        blocked = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Resize", "Upsample"]
        fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=blocked)
        onnx.save(fp16, dst)
        a, b = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
        print(f"  + {os.path.basename(dst)}  {a:.0f} MB -> {b:.0f} MB")
        return True
    except Exception as e:  # noqa: BLE001 — report and continue with the rest
        print(f"  ! {os.path.basename(src)}: {e}")
        if os.path.exists(dst):
            os.remove(dst)  # never leave a half-written model for the loader
        return False


def main() -> None:
    if len(sys.argv) > 1:
        convert(sys.argv[1])
        return
    print(f"skills dir: {SKILLS}")
    for rel, note in TARGETS:
        src = os.path.join(SKILLS, rel)
        if os.path.exists(src):
            print(f"{rel} — {note}")
            convert(src)
    print("done. delete a .fp16.onnx to revert that model to FP32.")


if __name__ == "__main__":
    main()
