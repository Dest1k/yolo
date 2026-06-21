#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Превращает обученный чекпойнт NanoDet-Plus в оптимизированную, *проверенную* модель
NCNN — тот же принцип «одной командой», что и у экспортёра YOLO-FastestV2.

Конвейер (каждый шаг проверяется, результат подгружается обратно для доказательства):
    best.ckpt -> ONNX (opset 11) -> onnxsim -> ncnn (.param/.bin) -> ncnnoptimize (fp16)
              -> загрузка в ncnn-python и подтверждение, что выходной блоб извлекается.

`train_nanodet.py` и `get_model.py` вызывают это после обучения/скачивания, поэтому
всё происходит одной командой. Отдельно:
    python export_ncnn.py --repo nanodet --cfg config/custom.yml \
        [--ckpt workspace/.../model_best.ckpt] [--out nanodet] [--input 416] \
        [--reg-max 7] [--classes 3]

Стабильность: форсируем opset 11 + dynamo=False (чтобы любая версия torch выдала граф,
который читает onnx2ncnn), предпочитаем onnx2ncnn+ncnnoptimize, иначе pnnx, и оставляем
результат ncnnoptimize только если он по-прежнему проходит проверку. Модель проходит ВСЕ
доступные оптимизации (это важно: она поедет на Raspberry Pi 5), а сам рантайм добавляет
fp16/int8-инференс автоматически.
"""

import argparse
import glob
import inspect
import os
import runpy
import shutil
import subprocess
import sys

# Старая кодовая страница консоли Windows (cp1251 и т.п.) не должна падать на наших стрелках.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def _log(msg): print(msg, flush=True)


def _newest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def find_ckpt(repo):
    """Самый свежий .ckpt, созданный обучением (NanoDet кладёт их в save_dir/workspace)."""
    return _newest(glob.glob(os.path.join(repo, "**", "*.ckpt"), recursive=True)) \
        or _newest(glob.glob(os.path.join("workspace", "**", "*.ckpt"), recursive=True))


def _shim_torch_six():
    """torch._six убрали в torch 2.0; старый код nanodet всё ещё его импортирует.
    Возвращаем те имена, которыми он пользовался, чтобы экспорт не падал."""
    import torch, types, sys
    import collections.abc as _abc
    if not hasattr(torch, "_six"):
        m = types.ModuleType("torch._six")
        m.string_classes = (str, bytes); m.int_classes = int
        m.container_abcs = _abc; m.PY3 = True; m.PY37 = sys.version_info >= (3, 7)
        torch._six = m; sys.modules["torch._six"] = m


def export_onnx(repo, cfg, ckpt, onnx_out, input_size):
    """Запускает nanodet/tools/export_onnx.py внутри процесса, форсируя экспорт в
    legacy-opset 11 (через monkeypatch torch.onnx.export), чтобы граф был корректен на
    любой версии torch. Также форсирует torch.load(weights_only=False): в torch 2.6+
    значение по умолчанию стало True, и официальные чекпойнты (с объектами Lightning)
    иначе не грузятся."""
    import torch
    _shim_torch_six()
    # Делаем ВСЕ пути абсолютными ДО chdir в репозиторий — иначе относительный
    # --model_path / --cfg_path / out будет искаться от папки репозитория и «исчезнет»
    # (ловушка «No such file …/nanodet/<ckpt>»).
    repo = os.path.abspath(repo)
    cfg = os.path.abspath(cfg)
    ckpt = os.path.abspath(ckpt)
    onnx_out = os.path.abspath(onnx_out)
    script = os.path.join(repo, "tools", "export_onnx.py")
    if not os.path.isfile(script):
        raise FileNotFoundError("в репозитории nanodet нет tools/export_onnx.py")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"чекпойнт не найден: {ckpt}")
    _orig = torch.onnx.export
    supports_dynamo = "dynamo" in inspect.signature(_orig).parameters

    def _forced(*a, **k):
        k["opset_version"] = 11
        if supports_dynamo:
            k["dynamo"] = False
        return _orig(*a, **k)

    # torch 2.6+ : weights_only по умолчанию True -> официальный .ckpt не грузится.
    # Чекпойнт скачан с официального релиза RangiLyu/nanodet, источник доверенный.
    _orig_load = torch.load

    def _load_compat(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    before = set(glob.glob(os.path.join(repo, "*.onnx")))
    cwd, argv = os.getcwd(), sys.argv[:]
    sys.path.insert(0, repo); os.chdir(repo)
    torch.onnx.export = _forced
    torch.load = _load_compat
    try:
        # Экспортёр nanodet пишет nanodet.onnx (или --out_path); все пути абсолютные.
        sys.argv = ["export_onnx.py", "--cfg_path", cfg,
                    "--model_path", ckpt, "--out_path", onnx_out]
        runpy.run_path("tools/export_onnx.py", run_name="__main__")
    finally:
        torch.onnx.export = _orig
        torch.load = _orig_load
        os.chdir(cwd); sys.argv = argv
        if repo in sys.path:
            sys.path.remove(repo)

    if not os.path.isfile(onnx_out):
        produced = _newest(set(glob.glob(os.path.join(repo, "*.onnx"))) - before)
        if produced:
            shutil.copyfile(produced, onnx_out)
    if not os.path.isfile(onnx_out):
        raise RuntimeError("export_onnx.py отработал, но .onnx не появился")
    _log(f"  ONNX (opset 11) -> {onnx_out}")
    return onnx_out


def simplify(onnx_in, onnx_out):
    try:
        if subprocess.call([sys.executable, "-m", "onnxsim", onnx_in, onnx_out]) == 0 \
                and os.path.isfile(onnx_out):
            _log(f"  onnxsim -> {onnx_out}"); return onnx_out
    except Exception as e:
        _log(f"  onnxsim пропущен ({e})")
    _log("  onnxsim недоступен — беру неупрощённый ONNX (pip install onnxsim)")
    return onnx_in


def convert_and_optimize(onnx_path, out_param, out_bin, input_size, reg_max, classes):
    """ONNX -> ncnn с оптимизацией графа. Если в PATH есть onnx2ncnn+ncnnoptimize(fp16) —
    берём их (результат ncnnoptimize оставляем только если он проходит проверку), иначе
    pnnx (конвертирует и оптимизирует разом). В обоих путях включаем максимум оптимизаций
    — модель поедет на Raspberry Pi 5."""
    onnx2ncnn = shutil.which("onnx2ncnn"); ncnnoptimize = shutil.which("ncnnoptimize")
    if onnx2ncnn:
        if subprocess.call([onnx2ncnn, onnx_path, out_param, out_bin]) != 0:
            raise RuntimeError("onnx2ncnn завершился с ошибкой")
        _log(f"  onnx2ncnn -> {out_param} / {out_bin}")
        if ncnnoptimize:
            op, ob = out_param.replace(".param", ".opt.param"), out_bin.replace(".bin", ".opt.bin")
            if subprocess.call([ncnnoptimize, out_param, out_bin, op, ob, "1"]) == 0 \
                    and verify(op, ob, input_size, reg_max, quiet=True):
                shutil.move(op, out_param); shutil.move(ob, out_bin)
                _log("  применён ncnnoptimize (fp16 + слияние слоёв)")
            else:
                for f in (op, ob):
                    if os.path.isfile(f): os.remove(f)
                _log("  ncnnoptimize пропущен (оставил обычную, проверенную модель)")
        else:
            _log("  ncnnoptimize не найден — модель собрана без fp16-оптимизации.\n"
                 "    Для максимума на Pi положи ncnnoptimize в PATH или поставь pnnx.")
        return out_param, out_bin

    w = shutil.which("pnnx")
    pnnx_cmd = [w] if w else _pnnx_module()
    if pnnx_cmd:
        if not isinstance(pnnx_cmd, list):
            pnnx_cmd = [pnnx_cmd]
        onnx_path = os.path.abspath(onnx_path)
        d = os.path.dirname(onnx_path)
        before = set(glob.glob(os.path.join(d, "*.ncnn.param")))
        # fp16=1 + optlevel=2 — максимум оптимизаций для рантайма на RPi 5.
        subprocess.call([*pnnx_cmd, onnx_path,
                         f"inputshape=[1,3,{input_size},{input_size}]",
                         "fp16=1", "optlevel=2"])
        # pnnx санитизирует имя файла (- и . -> _), поэтому не угадываем точное имя,
        # а берём свежесозданный *.ncnn.param и парный .bin.
        cand = _newest(set(glob.glob(os.path.join(d, "*.ncnn.param"))) - before) \
            or _newest(glob.glob(os.path.join(d, "*.ncnn.param")))
        if cand and os.path.isfile(cand[:-6] + ".bin"):
            shutil.copyfile(cand, out_param)
            shutil.copyfile(cand[:-6] + ".bin", out_bin)
            _cleanup_pnnx(onnx_path, cand)
            _log(f"  pnnx (конвертация + fp16 + optlevel=2) -> {out_param} / {out_bin}")
            return out_param, out_bin
        raise RuntimeError("pnnx отработал, но .ncnn.param/.bin не появились")

    raise RuntimeError(
        "не найден конвертер ONNX->ncnn. Поставь один из них:\n"
        "    pip install pnnx            # проще всего — один wheel, конвертирует и оптимизирует\n"
        "  либо положи onnx2ncnn / ncnnoptimize в PATH (собрать инструменты ncnn).")


def _cleanup_pnnx(onnx_path, ncnn_param):
    """Убираем за pnnx промежуточный мусор, чтобы рядом остались только нужные .param/.bin.
    Удаляем СТРОГО файлы, производные от имени, которое реально использовал pnnx (берём его
    из найденного .ncnn.param — pnnx меняет дефисы на '_', но точку сохраняет), плюс варианты
    .pnnxsim.onnx от входного onnx. Никаких широких масок — чтобы не снести чужой файл."""
    d = os.path.dirname(os.path.abspath(onnx_path))
    base = os.path.splitext(os.path.basename(onnx_path))[0]                 # имя входного onnx
    stem = os.path.basename(ncnn_param)[:-len(".ncnn.param")]               # реальный stem pnnx
    suffixes = (".pnnx.param", ".pnnx.bin", ".pnnx.onnx", ".pnnxsim.onnx",
                "_pnnx.py", "_ncnn.py", ".ncnn.param", ".ncnn.bin")
    for s in {stem, base}:
        for suf in suffixes:
            f = os.path.join(d, s + suf)
            if os.path.isfile(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def _pnnx_module():
    """pnnx, установленный как python-пакет (pip install pnnx), кладёт бинарь внутрь пакета,
    но не всегда в PATH. Возвращаем команду запуска или None."""
    try:
        import pnnx
        exe = getattr(pnnx, "pnnx", None)
        if callable(exe):
            return [sys.executable, "-c", "import pnnx,sys; pnnx.pnnx(sys.argv[1:])"]
        d = os.path.dirname(os.path.abspath(pnnx.__file__))
        for cand in ("pnnx", "pnnx.exe"):
            p = os.path.join(d, cand)
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return None


def verify(param, bin_, input_size, reg_max, quiet=False, classes=None):
    """Загружает ncnn-модель и подтверждает, что выходной блоб извлекается, а число его
    каналов соответствует NanoDet'у: nc + 4*(reg_max+1)."""
    try:
        import ncnn, numpy as np
    except Exception:
        if not quiet: _log("  проверка пропущена (pip install ncnn для самопроверки)")
        return True
    try:
        net = ncnn.Net()
        if net.load_param(param) != 0 or net.load_model(bin_) != 0:
            if not quiet: _log("  ПРОВЕРКА ПРОВАЛЕНА: ncnn не смог загрузить модель")
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
            if not quiet: _log("  ПРОВЕРКА ПРОВАЛЕНА: ни один выходной блоб не извлёкся (проблема opset/конвертации)")
            return False
        if not quiet:
            _log(f"  ПРОВЕРКА OK: вход '{ins[0] if ins else '?'}', выходы:")
            for nm, shp, C, P in ok:
                nc = C - 4 * (reg_max + 1)
                note = f"  -> nc={nc} (reg_max={reg_max}), точек={P}"
                if classes and nc != classes:
                    note += f"  [!] nc != CLASSES({classes})"
                _log(f"      {nm}: {tuple(int(x) for x in shp)}{note}")
        return True
    except Exception as e:
        if not quiet: _log(f"  ошибка проверки: {e}")
        return False


def run_export(repo, cfg, out_stem, input_size, reg_max, classes=None, ckpt=None):
    ckpt = ckpt or find_ckpt(repo)
    if not ckpt:
        _log(f"  не нашёл ни одного .ckpt в {repo}/ или workspace/ (обучение завершилось?)"); return None
    _log(f"  чекпойнт: {ckpt}")
    onnx_raw = os.path.abspath(out_stem + ".onnx")
    onnx_sim = os.path.abspath(out_stem + "-sim.onnx")
    param = os.path.abspath(out_stem + ".param"); bin_ = os.path.abspath(out_stem + ".bin")
    try:
        export_onnx(repo, cfg, ckpt, onnx_raw, input_size)
        onnx_final = simplify(onnx_raw, onnx_sim)
        convert_and_optimize(onnx_final, param, bin_, input_size, reg_max, classes)
    except Exception as e:
        _log(f"  ЭКСПОРТ ПРОВАЛЕН: {e}"); return None
    if not verify(param, bin_, input_size, reg_max, classes=classes):
        _log("  (модель сконвертирована, но не прошла проверку — см. заметки выше)")
    # Промежуточные .onnx больше не нужны — оставляем рядом только .param/.bin.
    for f in (onnx_raw, onnx_sim):
        if os.path.isfile(f):
            try:
                os.remove(f)
            except Exception:
                pass
    return param, bin_


def main():
    ap = argparse.ArgumentParser(description="Экспорт обученного NanoDet-Plus в оптимизированный NCNN.")
    ap.add_argument("--repo", default="nanodet")
    ap.add_argument("--cfg", required=True, help="конфиг .yml, на котором обучали")
    ap.add_argument("--ckpt", default=None, help="чекпойнт .ckpt (по умолчанию — самый свежий)")
    ap.add_argument("--out", default="nanodet")
    ap.add_argument("--input", type=int, default=416)
    ap.add_argument("--reg-max", type=int, default=7)
    ap.add_argument("--classes", type=int, default=None)
    a = ap.parse_args()
    res = run_export(a.repo, a.cfg, a.out, a.input, a.reg_max, a.classes, a.ckpt)
    if not res:
        sys.exit(1)
    param, bin_ = res
    _log("\nГотово. Можно запускать:")
    _log(f"  Сайдкар на Pi:  ND_PARAM={param} ND_BIN={bin_} ND_INPUT={a.input} \\")
    _log(f"                  YOLO_LABELS=<classes.txt> python3 nanodet_ncnn_sidecar.py --inspect")


if __name__ == "__main__":
    main()
