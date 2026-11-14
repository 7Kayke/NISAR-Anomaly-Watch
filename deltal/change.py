"""Contas da detecção de mudança (ver Apostila de cálculo, contas 6 a 12)."""
import numpy as np
from scipy import ndimage


def to_db(x):
    """Potência linear -> dB. Zero, negativo e NaN viram NaN."""
    x = np.asarray(x, dtype="float64")
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = 10.0 * np.log10(x[ok])
    return out


def from_db(db):
    return 10.0 ** (np.asarray(db, dtype="float64") / 10.0)


def boxcar_linear(x, size=5):
    """Média k×k em LINEAR, ignorando NaN (conta 7)."""
    x = np.asarray(x, dtype="float64")
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    s = ndimage.uniform_filter(filled, size=size, mode="nearest")
    c = ndimage.uniform_filter(valid.astype("float64"), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < 0.5] = np.nan
    return out


def log_ratio_db(before_lin, after_lin, size=5):
    """Razão logarítmica depois/antes em dB, com filtro igual nas duas datas (conta 10)."""
    return to_db(boxcar_linear(after_lin, size)) - to_db(boxcar_linear(before_lin, size))


def robust_z(values, mask=None, min_mad=0.1):
    """z robusto espacial: (x - mediana) / (1,4826 · MAD) (conta 11)."""
    v = np.asarray(values, dtype="float64")
    m = np.isfinite(v) if mask is None else (mask & np.isfinite(v))
    if m.sum() < 30:
        return np.full(v.shape, np.nan), np.nan, np.nan
    med = float(np.median(v[m]))
    mad = max(float(np.median(np.abs(v[m] - med))), min_mad)
    return (v - med) / (1.4826 * mad), med, mad


def min_area_filter(mask, min_pixels):
    """Remove grupos de pixels vizinhos menores que min_pixels (conta 12)."""
    mask = np.asarray(mask, dtype=bool)
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep_ids = np.where(sizes >= min_pixels)[0] + 1
    keep = np.isin(lab, keep_ids)
    return keep, [int(sizes[i - 1]) for i in keep_ids]


def nbr(nir, swir2):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - swir2) / (nir + swir2)


def ndvi(nir, red):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red) / (nir + red)
