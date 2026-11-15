"""Testes sintéticos do funil, do relevo e do consenso (sem rede)."""
from datetime import datetime

import numpy as np

from deltal.consensus import combine
from deltal.funnel import Funnel, coverage, pair_by_key
from deltal.terrain import radar_masks, look_azimuth


def test_funnel_percentages():
    f = Funnel("X", "x")
    f.add("catalogo", range(10)); f.add("cobre_area", range(8)); f.add("qualidade", range(4)); f.add("pareados", range(2))
    s = f.summary()
    assert [e["aproveitamento_pct"] for e in s["etapas"]] == [None, 80.0, 50.0, 50.0]
    assert s["aproveitamento_total_pct"] == 20.0


def test_coverage():
    poly = {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}
    assert coverage(poly, (0, 0, 1, 1)) == 1.0
    assert abs(coverage(poly, (0.5, 0, 1.5, 1)) - 0.5) < 1e-9


def test_pair_same_geometry_smallest_gap():
    ev = datetime(2026, 10, 16)
    items = [("A", datetime(2026, 10, 2)), ("A", datetime(2026, 10, 14)), ("A", datetime(2026, 10, 26)),
             ("B", datetime(2026, 10, 15)), ("B", datetime(2026, 11, 20))]
    k, a, b, gap = pair_by_key(items, ev, key=lambda i: i[0], date_of=lambda i: i[1])
    assert k == "A" and a[1].day == 14 and b[1].day == 26 and gap == 12.0
    assert pair_by_key(items[:2], ev, key=lambda i: i[0], date_of=lambda i: i[1]) is None


def test_look_sides():
    # ascendente: S1 olha para leste (~80°), NISAR olha para oeste (~260°)
    assert abs(look_azimuth("sentinel-1", "ascending") - 80) < 1e-9
    assert abs(look_azimuth("nisar", "A") - 260) < 1e-9


def test_radar_masks_ramp():
    # rampa íngreme que sobe para leste (60° de declividade)
    x = np.arange(50) * 20.0
    dem = np.tile(x * np.tan(np.radians(60)), (50, 1))
    lay_s1, sha_s1, _ = radar_masks(dem, 20.0, "sentinel-1", "ascending")
    lay_ni, sha_ni, _ = radar_masks(dem, 20.0, "nisar", "A")
    # S1 olha para leste: a encosta sobe ao se afastar do satélite -> encosta de frente -> layover
    assert lay_s1[5:-5, 5:-5].all() and not sha_s1.any()
    # NISAR olha para oeste: a encosta cai ao se afastar -> encosta de costas, 60° > 90-37 -> sombra
    assert sha_ni[5:-5, 5:-5].all() and not lay_ni.any()
    flat = np.zeros((20, 20))
    l, s, _ = radar_masks(flat, 20.0, "nisar", "D")
    assert not l.any() and not s.any()


def test_consensus_votes_and_availability():
    shape = (60, 60)
    burn = np.zeros(shape, bool); burn[10:40, 10:40] = True
    noise = np.zeros(shape, bool); noise[50:55, 50:55] = True
    all_ok = np.ones(shape, bool)
    half = np.ones(shape, bool); half[:, 30:] = False
    c = combine({"a": (burn, all_ok), "b": (burn | noise, all_ok), "c": (burn, half)}, min_votes=2,
                pixel_area_m2=400, min_area_m2=4000)
    assert c["consenso"][10:40, 10:40].all()
    assert not c["consenso"][50:55, 50:55].any()            # só uma fonte viu
    assert c["confianca"][20, 20] == 1.0 and c["disponiveis"][20, 35] == 2
