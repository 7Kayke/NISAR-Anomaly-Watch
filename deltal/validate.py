# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Validação: matriz de confusão e métricas (conta 20)."""
import numpy as np


def confusion(pred, truth, valid=None):
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
    v = np.ones(pred.shape, bool) if valid is None else np.asarray(valid, bool)
    vp = int(np.sum(pred & truth & v))
    fp = int(np.sum(pred & ~truth & v))
    fn = int(np.sum(~pred & truth & v))
    vn = int(np.sum(~pred & ~truth & v))
    return vp, fp, fn, vn


def metrics(pred, truth, valid=None, pixel_area_m2=None):
    vp, fp, fn, vn = confusion(pred, truth, valid)
    def div(a, b):
        return float(a) / b if b else float("nan")
    p = div(vp, vp + fp)
    s = div(vp, vp + fn)
    f1 = div(2 * p * s, p + s) if (p == p and s == s and (p + s) > 0) else float("nan")
    out = {
        "VP": vp, "FP": fp, "FN": fn, "VN": vn,
        "precisao": p, "sensibilidade": s, "f1": f1,
        "iou": div(vp, vp + fp + fn),
        "falso_alarme": div(fp, fp + vn),
    }
    if pixel_area_m2:
        out["area_detectada_ha"] = (vp + fp) * pixel_area_m2 / 1e4
        out["area_real_ha"] = (vp + fn) * pixel_area_m2 / 1e4
    return out
