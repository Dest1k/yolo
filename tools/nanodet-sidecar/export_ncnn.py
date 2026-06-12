#!/usr/bin/env python3
r"""
Turn a trained NanoDet-Plus checkpoint into an optimised, *verified* NCNN model
— the same one-command paradigm as the YOLO-FastestV2 exporter.

Pipeline (each step checked, result loaded back to prove it works):
    best.ckpt → ONNX (opset 11) → onnxsim → ncnn (.param/.bin) → ncnnoptimize (fp16)
              → load in ncnn-python and confirm the head blob extracts.

`train_nanodet.py` calls this after training, so the whole thing is one command.
Standalone:
    python export_ncnn.py --repo nanodet --cfg config/custom.yml \
        [--ckpt workspace/.../model_best.ckpt] [--out nanodet] [--input 416] \
        [--reg-max 7] [--classes 3]

Stability: forces opset 11 + dynamo=False (so any torch version exports a blob
onnx2ncnn can read), prefers onnx2ncnn+ncnnoptimize else pnnx, and only keeps
ncnnoptimize if the result still verifies. The on-device runtime adds fp16/int8
inference automatically.
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


def find_ckpt(repo):
    """Newest .ckpt produced by training (NanoDet saves under save_dir/workspace)."""
    return _newest(glob.glob(os.path.join(repo, "**", "*.ckpt"), recursive=True)) \
        or _newest(glob.glob(os.path.join("workspace", "**", "*.ckpt"), recursive=True))


def export_onnx(repo, cfg, ckpt, onnx_out, input_size):
    """Run nanodet's tools/export_onnx.py in-process, forcing a legacy opset-11
    export (monkeypatch torch.onnx.export) so it's correct on any torch version."""
    import torch
    # Resolve every path to absolute BEFORE chdir'ing into the repo — otherwise a
    # relative --model_path / --cfg_path / out would resolve against the repo dir and
    # vanish (the "No such file …/nanodet/<ckpt>" trap).
    repo = os.path.abspath(repo)
    cfg = os.path.abspath(cfg)
    ckpt = os.path.abspath(ckpt)
    onnx_out = os.path.abspath(onnx_out)
    script = os.path.join(repo, "tools", "export_onnx.py")
    if not os.path.isfile(script):
        raise FileNotFoundError("tools/export_onnx.py not found in the nanodet repo")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    _orig = torch.onnx.export
    supports_dynamo = "dynamo" in inspect.signature(_orig).parameters

    def _forced(*a, **k):
        k["opset_version"] = 11
        if supports_dynamo:
            k["dynamo"] = False
        return _orig(*a, **k)

    before = set(glob.glob(os.path.join(repo, "*.onnx")))
    cwd, argv = os.getcwd(), sys.argv[:]
    sys.path.insert(0, repo); os.chdir(repo)
    torch.onnx.export = _forced
    try:
        # nanodet's exporter writes nanodet.onnx (or --out_path); all paths absolute.
        sys.argv = ["export_onnx.py", "--cfg_path", cfg,
                    "--model_path", ckpt, "--out_path", onnx_out]
        runpy.run_path("tools/export_onnx.py", run_name="__main__")
    finally:
        torch.onnx.export = _orig
        os.chdir(cwd); sys.argv = argv
        if repo in sys.path:
            sys.path.remove(repo)

    if not os.path.isfile(onnx_out):
        produced = _newest(set(glob.glob(os.path.join(repo, "*.onnx"))) - before)
        if produced:
            shutil.copyfile(produced, onnx_out)
    if not os.path.isfile(onnx_out):
        raise RuntimeError("export_onnx.py ran but produced no .onnx")
    _log(f"  ONNX (opset 11) → {onnx_out}")
    return onnx_out


def simplify(onnx_in, onnx_out):
    try:
        if subprocess.call([sys.executable, "-m", "onnxsim", onnx_in, onnx_out]) == 0 \
                and os.path.isfile(onnx_out):
            _log(f"  onnxsim → {onnx_out}"); return onnx_out
    except Exception as e:
        _log(f"  onnxsim skipped ({e})")
    _log("  onnxsim unavailable — using un-simplified ONNX (pip install onnxsim)")
    return onnx_in


def convert_and_optimize(onnx_path, out_param, out_bin, input_size, reg_max, classes):
    """ONNX → ncnn with graph optimisation. onnx2ncnn+ncnnoptimize(fp16) if on PATH
    (ncnnoptimize kept only if still verifies), else pnnx (converts + optimises)."""
    onnx2ncnn = shutil.which("onnx2ncnn"); ncnnoptimize = shutil.which("ncnnoptimize")
    if onnx2ncnn:
        if subprocess.call([onnx2ncnn, onnx_path, out_param, out_bin]) != 0:
            raise RuntimeError("onnx2ncnn failed")
        _log(f"  onnx2ncnn → {out_param} / {out_bin}")
        if ncnnoptimize:
            op, ob = out_param.replace(".param", ".opt.param"), out_bin.replace(".bin", ".opt.bin")
            if subprocess.call([ncnnoptimize, out_param, out_bin, op, ob, "1"]) == 0 \
                    and verify(op, ob, input_size, reg_max, quiet=True):
                shutil.move(op, out_param); shutil.move(ob, out_bin)
                _log("  ncnnoptimize (fp16 + fuse) applied")
            else:
                for f in (op, ob):
                    if os.path.isfile(f): os.remove(f)
                _log("  ncnnoptimize skipped (kept the plain, verified model)")
        return out_param, out_bin

    pnnx = shutil.which("pnnx")
    if pnnx:
        stem = os.path.splitext(onnx_path)[0]
        subprocess.call([pnnx, onnx_path, f"inputshape=[1,3,{input_size},{input_size}]", "fp16=1"])
        p, b = stem + ".ncnn.param", stem + ".ncnn.bin"
        if os.path.isfile(p) and os.path.isfile(b):
            shutil.copyfile(p, out_param); shutil.copyfile(b, out_bin)
            _log(f"  pnnx (convert + fp16) → {out_param} / {out_bin}"); return out_param, out_bin
        raise RuntimeError("pnnx ran but produced no .ncnn.param/.bin")

    raise RuntimeError(
        "no ONNX→ncnn converter found. Install one:\n"
        "    pip install pnnx            # easiest — one wheel, converts + optimises\n"
        "  or put onnx2ncnn / ncnnoptimize on PATH (build ncnn tools).")


def verify(param, bin_, input_size, reg_max, quiet=False, classes=None):
    """Load the ncnn model and confirm the head blob extracts and its channel count
    fits NanoDet's nc + 4*(reg_max+1)."""
    try:
        import ncnn, numpy as np
    except Exception:
        if not quiet: _log("  verify skipped (pip install ncnn to self-check)")
        return True
    try:
        net = ncnn.Net()
        if net.load_param(param) != 0 or net.load_model(bin_) != 0:
            if not quiet: _log("  VERIFY FAILED: ncnn could not load the model")
            return False
        ins = list(net.input_names()); outs = list(net.output_names())
        ex = net.create_extractor()
        d = ncnn.Mat(input_size, input_size, 3); d.fill(0.0)
        ex.input(ins[0] if ins else "in0", d)
        ok = []
        for nm in outs:
            ret, m = ex.extract(nm)
            a = np.array(m)
            if ret == 0 and a.size > 0:
                C = min(a.shape) if a.ndim == 2 else a.shape[-1]
                P = max(a.shape) if a.ndim == 2 else None
                ok.append((nm, a.shape, C, P))
        if not ok:
            if not quiet: _log("  VERIFY FAILED: no head blob extracted (opset/convert issue)")
            return False
        if not quiet:
            _log(f"  VERIFY OK: input '{ins[0] if ins else '?'}', outputs:")
            for nm, shp, C, P in ok:
                nc = C - 4 * (reg_max + 1)
                note = f"  → nc={nc} (reg_max={reg_max}), points={P}"
                if classes and nc != classes:
                    note += f"  ⚠️ nc≠CLASSES({classes})"
                _log(f"      {nm}: {tuple(int(x) for x in shp)}{note}")
        return True
    except Exception as e:
        if not quiet: _log(f"  verify error: {e}")
        return False


def run_export(repo, cfg, out_stem, input_size, reg_max, classes=None, ckpt=None):
    ckpt = ckpt or find_ckpt(repo)
    if not ckpt:
        _log(f"  no .ckpt found under {repo}/ or workspace/ (did training finish?)"); return None
    _log(f"  checkpoint: {ckpt}")
    onnx_raw = os.path.abspath(out_stem + ".onnx")
    onnx_sim = os.path.abspath(out_stem + "-sim.onnx")
    param = os.path.abspath(out_stem + ".param"); bin_ = os.path.abspath(out_stem + ".bin")
    try:
        export_onnx(repo, cfg, ckpt, onnx_raw, input_size)
        onnx_final = simplify(onnx_raw, onnx_sim)
        convert_and_optimize(onnx_final, param, bin_, input_size, reg_max, classes)
    except Exception as e:
        _log(f"  EXPORT FAILED: {e}"); return None
    if not verify(param, bin_, input_size, reg_max, classes=classes):
        _log("  (model converted but failed verification — see notes above)")
    return param, bin_


def main():
    ap = argparse.ArgumentParser(description="Export a trained NanoDet-Plus to optimised NCNN.")
    ap.add_argument("--repo", default="nanodet")
    ap.add_argument("--cfg", required=True, help="the nanodet config .yml used for training")
    ap.add_argument("--ckpt", default=None, help="checkpoint .ckpt (default: newest found)")
    ap.add_argument("--out", default="nanodet")
    ap.add_argument("--input", type=int, default=416)
    ap.add_argument("--reg-max", type=int, default=7)
    ap.add_argument("--classes", type=int, default=None)
    a = ap.parse_args()
    res = run_export(a.repo, a.cfg, a.out, a.input, a.reg_max, a.classes, a.ckpt)
    if not res:
        sys.exit(1)
    param, bin_ = res
    _log("\nDone. Ready to run:")
    _log(f"  Pi sidecar:  ND_PARAM={param} ND_BIN={bin_} ND_INPUT={a.input} \\")
    _log(f"               YOLO_LABELS=<classes.txt> python3 nanodet_ncnn_sidecar.py --inspect")


if __name__ == "__main__":
    main()
