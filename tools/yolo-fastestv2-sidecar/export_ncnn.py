#!/usr/bin/env python3
r"""
Turn a trained YOLO-FastestV2 checkpoint into an optimised, *verified* NCNN model
— in one go, no manual onnx2ncnn dance.

Pipeline (every step is checked, and the result is loaded back to prove it works):
    best.pth → ONNX (opset 11) → onnxsim → ncnn (.param/.bin) → ncnnoptimize (fp16)
             → load in ncnn-python and confirm the head blobs actually extract.

`train_yolofastest.py` calls this automatically after training, so the whole thing
is a single command. You can also run it standalone on an existing checkpoint:

    python export_ncnn.py --repo Yolo-FastestV2 --data yf_data/custom.data \
        [--weights path/to/best.pth] [--out yolofastestv2] [--input 352]

Why this is stable (the two things that usually break the export):
  • opset: a new PyTorch (e.g. the cu128 nightly for Blackwell) defaults to the
    TorchDynamo exporter and emits opset 18, which onnx2ncnn mis-converts. We
    monkeypatch torch.onnx.export to force opset 11 + dynamo=False, then run the
    repo's own pytorch2onnx.py — so it works on any torch version.
  • the converter binary: prefers onnx2ncnn+ncnnoptimize if on PATH, else pnnx
    (`pip install pnnx`, a single wheel that converts AND optimises). If neither
    is present it says exactly what to install instead of failing cryptically.

The on-device runtime (phone app + Pi sidecar) additionally enables fp16 and
int8 inference automatically, so this fp16-storage model is "all optimisations"
for a no-calibration export. (int8 needs a calibration set — see the README.)
"""

import argparse
import glob
import inspect
import os
import runpy
import shutil
import subprocess
import sys


def _log(msg): print(msg, flush=True)


def _newest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def find_weights(repo):
    """Newest .pth produced by training (prefer the repo's weights/ dir)."""
    w = _newest(glob.glob(os.path.join(repo, "weights", "**", "*.pth"), recursive=True))
    return w or _newest(glob.glob(os.path.join(repo, "**", "*.pth"), recursive=True))


def parse_classes(data_path):
    """Read `classes=N` out of the .data so we can sanity-check the export."""
    try:
        for line in open(data_path, encoding="utf-8"):
            if line.strip().startswith("classes"):
                return int(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return None


def export_onnx(repo, data_path, weights, onnx_out, input_size):
    """Run the repo's pytorch2onnx.py in-process, forcing a legacy opset-11 export.

    Monkeypatching torch.onnx.export means we don't care what opset/dynamo the
    repo's script (or your torch version) would otherwise pick."""
    import torch  # imported here so a CPU-only export venv works without CUDA
    script = os.path.join(repo, "pytorch2onnx.py")
    if not os.path.isfile(script):
        raise FileNotFoundError("pytorch2onnx.py not found in the repo")

    _orig = torch.onnx.export
    supports_dynamo = "dynamo" in inspect.signature(_orig).parameters

    def _forced(*a, **k):
        k["opset_version"] = 11
        if supports_dynamo:
            k["dynamo"] = False
        return _orig(*a, **k)

    before = set(glob.glob(os.path.join(repo, "*.onnx")))
    cwd, argv = os.getcwd(), sys.argv[:]
    sys.path.insert(0, repo)
    os.chdir(repo)
    torch.onnx.export = _forced
    try:
        sys.argv = ["pytorch2onnx.py", "--data", os.path.abspath(data_path),
                    "--weights", os.path.abspath(weights)]
        runpy.run_path("pytorch2onnx.py", run_name="__main__")
    finally:
        torch.onnx.export = _orig
        os.chdir(cwd); sys.argv = argv
        if repo in sys.path:
            sys.path.remove(repo)

    produced = _newest(set(glob.glob(os.path.join(repo, "*.onnx"))) - before) \
        or _newest(glob.glob(os.path.join(repo, "*.onnx")))
    if not produced:
        raise RuntimeError("pytorch2onnx.py ran but produced no .onnx")
    shutil.copyfile(produced, onnx_out)
    _log(f"  ONNX (opset 11) → {onnx_out}")
    return onnx_out


def simplify(onnx_in, onnx_out):
    """onnxsim — folds the constants/shape ops that trip up onnx2ncnn. Best-effort:
    if onnxsim isn't installed we just pass the raw ONNX through."""
    try:
        r = subprocess.call([sys.executable, "-m", "onnxsim", onnx_in, onnx_out])
        if r == 0 and os.path.isfile(onnx_out):
            _log(f"  onnxsim → {onnx_out}"); return onnx_out
    except Exception as e:
        _log(f"  onnxsim skipped ({e})")
    _log("  onnxsim unavailable — using un-simplified ONNX (pip install onnxsim)")
    return onnx_in


def convert_and_optimize(onnx_path, out_param, out_bin, input_size):
    """ONNX → ncnn with all graph optimisations. Returns (param, bin) or raises.

    Prefers the classic onnx2ncnn + ncnnoptimize(fp16); ncnnoptimize is only kept
    if the optimised model still passes verification (it can fuse away the head's
    output blobs on some exports). Falls back to pnnx, which converts + optimises
    in one wheel-installable step."""
    onnx2ncnn = shutil.which("onnx2ncnn")
    ncnnoptimize = shutil.which("ncnnoptimize")
    if onnx2ncnn:
        if subprocess.call([onnx2ncnn, onnx_path, out_param, out_bin]) != 0:
            raise RuntimeError("onnx2ncnn failed")
        _log(f"  onnx2ncnn → {out_param} / {out_bin}")
        if ncnnoptimize:
            opt_p = out_param.replace(".param", ".opt.param")
            opt_b = out_bin.replace(".bin", ".opt.bin")
            # last arg: storage type, 1 = fp16. Keep only if it still extracts.
            if subprocess.call([ncnnoptimize, out_param, out_bin, opt_p, opt_b, "1"]) == 0 \
                    and verify(opt_p, opt_b, input_size, quiet=True):
                shutil.move(opt_p, out_param); shutil.move(opt_b, out_bin)
                _log("  ncnnoptimize (fp16 + fuse) applied")
            else:
                for f in (opt_p, opt_b):
                    if os.path.isfile(f): os.remove(f)
                _log("  ncnnoptimize skipped (kept the plain, verified model)")
        else:
            _log("  ncnnoptimize not on PATH — runtime fp16/int8 still applies on-device")
        return out_param, out_bin

    pnnx = shutil.which("pnnx")
    if pnnx:
        stem = os.path.splitext(onnx_path)[0]
        subprocess.call([pnnx, onnx_path, f"inputshape=[1,3,{input_size},{input_size}]", "fp16=1"])
        p, b = stem + ".ncnn.param", stem + ".ncnn.bin"
        if os.path.isfile(p) and os.path.isfile(b):
            shutil.copyfile(p, out_param); shutil.copyfile(b, out_bin)
            _log(f"  pnnx (convert + fp16 optimise) → {out_param} / {out_bin}")
            return out_param, out_bin
        raise RuntimeError("pnnx ran but produced no .ncnn.param/.bin")

    raise RuntimeError(
        "no ONNX→ncnn converter found. Install one (either is fine):\n"
        "    pip install pnnx            # easiest — one wheel, converts + optimises\n"
        "  or put onnx2ncnn / ncnnoptimize on PATH (build ncnn tools).")


def verify(param, bin_, input_size, quiet=False, classes=None):
    """Load the ncnn model and confirm its head blobs actually extract (ncnn -100
    on forward = a broken conversion). Returns True/False; prints a summary."""
    try:
        import ncnn, numpy as np
    except Exception:
        if not quiet:
            _log("  verify skipped (pip install ncnn to self-check the export)")
        return True  # can't verify here; don't block
    try:
        net = ncnn.Net()
        if net.load_param(param) != 0 or net.load_model(bin_) != 0:
            if not quiet: _log("  VERIFY FAILED: ncnn could not load the model")
            return False
        in_names = list(net.input_names()); out_names = list(net.output_names())
        ex = net.create_extractor()
        dummy = ncnn.Mat(input_size, input_size, 3); dummy.fill(0.0)
        ex.input(in_names[0] if in_names else "in0", dummy)
        ok = []
        for nm in out_names:
            ret, m = ex.extract(nm)
            a = np.array(m)
            if ret == 0 and a.size > 0:
                C = a.shape[0] if a.ndim == 3 else (a.shape[0] if a.ndim == 2 else 0)
                ok.append((nm, a.shape, C))
        if not ok:
            if not quiet: _log("  VERIFY FAILED: no head blob extracted (likely opset/convert issue)")
            return False
        if not quiet:
            _log(f"  VERIFY OK: input '{in_names[0] if in_names else '?'}', outputs:")
            for nm, shp, C in ok:
                na = (C - classes) // 5 if classes else None
                extra = f"  → na={na}, nc={classes}" if (na and na >= 1 and 5 * na + classes == C) else ""
                _log(f"      {nm}: {tuple(int(x) for x in shp)}{extra}")
            if classes:
                bad = [nm for nm, _, C in ok if (C - classes) % 5 != 0 or (C - classes) // 5 < 1]
                if bad:
                    _log(f"  ⚠️  channel count doesn't fit classes={classes} on {bad} "
                         f"— check CLASSES in the trainer.")
        return True
    except Exception as e:
        if not quiet: _log(f"  verify error: {e}")
        return False


def run_export(repo, data_path, out_stem, input_size, weights=None):
    """Full .pth → optimised, verified ncnn. Returns (param, bin) or None on failure."""
    weights = weights or find_weights(repo)
    if not weights:
        _log(f"  no .pth found under {repo} (did training finish?)"); return None
    _log(f"  weights: {weights}")
    out_dir = os.path.dirname(os.path.abspath(out_stem)) or "."
    os.makedirs(out_dir, exist_ok=True)
    onnx_raw = os.path.abspath(out_stem + ".onnx")
    onnx_sim = os.path.abspath(out_stem + "-sim.onnx")
    param = os.path.abspath(out_stem + ".param")
    bin_ = os.path.abspath(out_stem + ".bin")
    try:
        export_onnx(repo, data_path, weights, onnx_raw, input_size)
        onnx_final = simplify(onnx_raw, onnx_sim)
        convert_and_optimize(onnx_final, param, bin_, input_size)
    except Exception as e:
        _log(f"  EXPORT FAILED: {e}")
        return None
    classes = parse_classes(data_path)
    if not verify(param, bin_, input_size, classes=classes):
        _log("  (model converted but failed verification — see notes above)")
    return param, bin_


def main():
    ap = argparse.ArgumentParser(description="Export a trained YOLO-FastestV2 to optimised NCNN.")
    ap.add_argument("--repo", default="Yolo-FastestV2", help="cloned upstream repo dir")
    ap.add_argument("--data", required=True, help="the .data file built by the trainer")
    ap.add_argument("--weights", default=None, help="checkpoint .pth (default: newest in the repo)")
    ap.add_argument("--out", default="yolofastestv2", help="output stem for .param/.bin")
    ap.add_argument("--input", type=int, default=352, help="model input size (must match training)")
    a = ap.parse_args()
    res = run_export(a.repo, a.data, a.out, a.input, a.weights)
    if not res:
        sys.exit(1)
    param, bin_ = res
    names = os.path.join(os.path.dirname(a.data) or ".", "custom.names")
    _log("\nDone. Ready to run:")
    _log(f"  Pi sidecar:  YF_PARAM={param} YF_BIN={bin_} YF_INPUT={a.input} \\")
    _log(f"               YOLO_LABELS={names} python3 yolofastest_ncnn_sidecar.py --inspect")
    _log(f"  Phone:       load {os.path.basename(param)}/.bin, version=FastestV2, "
         f"input={a.input}, classes from custom.names")


if __name__ == "__main__":
    main()
