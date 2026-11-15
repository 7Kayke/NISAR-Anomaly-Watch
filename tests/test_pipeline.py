# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Testes com dados sintéticos: o detector precisa achar o que foi plantado e nada além."""
import json

import numpy as np

from deltal.change import to_db, from_db, robust_z, min_area_filter
from deltal.grid import make_grid
from deltal.layers import compute_layers
from deltal.validate import metrics
from deltal.sources_nisar import parse_name
from deltal import report

RNG = np.random.default_rng(42)


def speckle(mean_lin, looks=8):
    return mean_lin * RNG.gamma(looks, 1.0 / looks, size=mean_lin.shape)


def scene(shape=(200, 200)):
    co = np.full(shape, from_db(-7.0))    # floresta: HH ~ -7 dB
    x = np.full(shape, from_db(-12.0))    # HV ~ -12 dB
    return co, x


def test_db_roundtrip():
    v = np.array([0.05, 1.0, 0.0, -1.0, np.nan])
    d = to_db(v)
    assert abs(d[0] + 13.0103) < 1e-3 and d[1] == 0
    assert np.isnan(d[2:]).all()
    assert abs(from_db(d[0]) - 0.05) < 1e-9


def test_robust_z_and_min_area():
    v = RNG.normal(0, 1, (100, 100))
    z, med, mad = robust_z(v)
    assert abs(med) < 0.1 and 0.5 < mad < 0.8
    m = np.zeros((50, 50), bool)
    m[10:20, 10:20] = True
    m[40, 40] = True
    kept, sizes = min_area_filter(m, 10)
    assert sizes == [100] and not kept[40, 40]


def test_detects_planted_burn_and_flood():
    co0, x0 = scene()
    co1, x1 = co0.copy(), x0.copy()
    burn = np.zeros(co0.shape, bool); burn[40:80, 40:90] = True
    flood = np.zeros(co0.shape, bool); flood[130:170, 120:170] = True
    x1[burn] = from_db(-18.0); co1[burn] = from_db(-8.0)        # HV -6 dB, HH -1 dB
    co1[flood] = from_db(-22.0); x1[flood] = from_db(-27.0)     # água lisa
    res = compute_layers(speckle(co0), speckle(co1), speckle(x0), speckle(x1), pixel_area_m2=400)
    m_burn = metrics(res["vegetacao"], burn)
    m_flood = metrics(res["agua"], flood)
    assert m_burn["iou"] > 0.7, m_burn
    assert m_flood["iou"] > 0.7, m_flood
    # a cheia não pode ser chamada de fogo em massa
    assert (res["vegetacao"] & flood).sum() < 0.2 * flood.sum()


def test_no_change_gives_no_alerts():
    co0, x0 = scene()
    res = compute_layers(speckle(co0), speckle(co0), speckle(x0), speckle(x0), pixel_area_m2=400)
    assert res["vegetacao"].sum() == 0 and res["agua"].sum() == 0


def test_global_shift_is_ignored():
    co0, x0 = scene()
    burn = np.zeros(co0.shape, bool); burn[50:90, 50:90] = True
    co1, x1 = co0 * from_db(1.0), x0 * from_db(1.0)   # cena inteira +1 dB (vento, calibração)
    x1[burn] = from_db(-18.0)
    res = compute_layers(speckle(co0), speckle(co1), speckle(x0), speckle(x1), pixel_area_m2=400)
    assert metrics(res["vegetacao"], burn)["iou"] > 0.7


def test_grid_and_names(tmp_path):
    g = make_grid([-117.72, 33.42, -117.58, 33.53], 20)
    assert g.epsg == 32611 and 600 < g.width < 700 and 600 < g.height < 650
    info = parse_name("NISAR_L2_PR_GCOV_034_034_A_018_4005_DHDH_A_20261026T130749_20261026T130824_P05036_N_F_J_001")
    assert (info["ciclo"], info["pista"], info["direcao"], info["quadro"], info["crid"]) == (34, 34, "A", 18, "P05036")
    f = report.write_ficha(tmp_path, {"a": np.float64(1.23456), "b": [1, 2], "c": float("nan")})
    assert json.loads((tmp_path / "ficha_tecnica.json").read_text())["c"] is None
    assert f["a"] == 1.2346
