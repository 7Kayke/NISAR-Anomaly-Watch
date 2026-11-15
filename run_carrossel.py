# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Carrossel do Paiaguás: para CADA data em que o NISAR passou, a imagem do Planetary Computer mais próxima,
no MESMO recorte (mesma grade de 20 m) e com a MESMA escala de cor em todas as datas, para dar para comparar.

Uso:
    python run_carrossel.py --case cases/paiaguas.json --serie docs/results/paiaguas --out docs/results/paiaguas/carrossel

Fontes (Microsoft Planetary Computer, STAC):
  sentinel-2-l2a   cor natural e falsa cor de queimada (SWIR2, NIR, vermelho)   janela ±5 dias, nuvem na área <= 40 %
  landsat-c2-l2    cor natural e temperatura da superfície (banda térmica)      janela ±8 dias, nuvem na área <= 40 %
  sentinel-1-rtc   radar banda C (VV, VH): o "primo" do NISAR, também vê através de nuvem   janela ±6 dias
Saída:
  <fonte>_<AAAAMMDD do NISAR>_<produto>.png   imagem crua da área (sem moldura), para alinhar com o NISAR no app
  carrossel.json   por data do NISAR: qual cena foi usada, a quantos dias, nuvem e os números (NDVI, NBR, queimada, água, °C, VH)
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deltal import collect, report
from deltal.grid import make_grid
from deltal.sources_pc import _search, _load, S2_CLOUD_SCL, S2_WATER_SCL

WIN = {"s2": 5, "landsat": 8, "s1": 6}
MAX_CLOUD = 0.4


def log(m):
    print(m, flush=True)


def nisar_dates(serie):
    """Todas as datas em que alguma geometria do NISAR passou, com quais geometrias."""
    out = {}
    for g in sorted(Path(serie).iterdir()):
        app = g / "app" / "app.json"
        if app.exists():
            for d in json.loads(app.read_text(encoding="utf-8"))["datas"]:
                out.setdefault(d, []).append(g.name)
    return dict(sorted(out.items()))


def save(path, rgb):
    plt.imsave(path, np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1))


def fixed(x, lo, hi, gamma=1.0):
    return np.clip((x - lo) / (hi - lo), 0, 1) ** gamma


def nearest(acqs, day, days):
    d0 = datetime.fromisoformat(day)
    c = [(abs((a.dt.replace(tzinfo=None) - d0).days), a) for a in acqs]
    return sorted([x for x in c if x[0] <= days], key=lambda x: x[0])


def med(x, m=None):
    v = x[np.isfinite(x)] if m is None else x[m & np.isfinite(x)]
    return round(float(np.median(v)), 3) if v.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--serie", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=20)
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(case["bbox"], a.res)
    qg = make_grid(grid.bbox, 100)
    dates = nisar_dates(a.serie)
    log(f"{len(dates)} datas do NISAR: {', '.join(dates)}")
    t0 = (datetime.fromisoformat(min(dates)) - timedelta(days=10)).date().isoformat()
    t1 = (datetime.fromisoformat(max(dates)) + timedelta(days=10)).date().isoformat()

    log("buscando no Planetary Computer...")
    s2 = collect._cover(collect._acquisitions(_search("sentinel-2-l2a", grid.bbox, t0, t1, {"eo:cloud_cover": {"lt": 90}})), grid.bbox)
    ls = [i for i in _search("landsat-c2-l2", grid.bbox, t0, t1, {"eo:cloud_cover": {"lt": 90}}) if i.properties.get("platform") in ("landsat-8", "landsat-9")]
    ls = collect._cover(collect._acquisitions(ls), grid.bbox)
    okey = lambda it: (it.properties.get("sat:relative_orbit"), it.properties.get("sat:orbit_state"))
    s1 = collect._cover(collect._acquisitions(_search("sentinel-1-rtc", grid.bbox, t0, t1), okey), grid.bbox)
    log(f"  Sentinel-2 {len(s2)} cenas · Landsat {len(ls)} · Sentinel-1 {len(s1)}")

    cloud = {}
    def s2_cloud(acq):
        if acq.day not in cloud:
            scl = _load(acq.items, ["SCL"], qg, resampling="nearest")["SCL"].values[0]
            cloud[acq.day] = float(np.isin(scl, list(S2_CLOUD_SCL)).mean())
        return cloud[acq.day]
    def ls_cloud(acq):
        k = "L" + acq.day
        if k not in cloud:
            qa = _load(acq.items, ["qa_pixel"], qg, resampling="nearest")["qa_pixel"].values[0].astype("int64")
            cl = np.zeros(qa.shape, bool)
            for b in collect.LANDSAT_CLOUD_BITS:
                cl |= ((qa >> b) & 1) == 1
            cloud[k] = float(cl.mean())
        return cloud[k]

    rec, done, nbr0 = [], {}, None
    for day, geos in dates.items():
        tag = day.replace("-", "")
        r = {"data_nisar": day, "geometrias": geos}
        log(f"[{day}] NISAR {', '.join(geos)}")
        # Sentinel-2: a mais próxima com pouca nuvem dentro da área
        try:
            pick = next(((dd, x, c) for dd, x in nearest(s2, day, WIN["s2"]) for c in [s2_cloud(x)] if c <= MAX_CLOUD), None)
            if pick:
                dd, acq, c = pick
                key = "s2" + acq.day
                if key not in done:
                    refl, clear, water = collect.sentinel2_load(acq, grid)
                    nat = np.dstack([fixed(refl["B04"], 0, 0.22, 0.8), fixed(refl["B03"], 0, 0.20, 0.8), fixed(refl["B02"], 0, 0.18, 0.8)])
                    brn = np.dstack([fixed(refl["B12"], 0, 0.40), fixed(refl["B08"], 0, 0.45), fixed(refl["B04"], 0, 0.22)])
                    f1, f2 = f"s2_{acq.day.replace('-', '')}_natural.png", f"s2_{acq.day.replace('-', '')}_queimada.png"
                    save(out / f1, nat); save(out / f2, brn)
                    ndvi = (refl["B08"] - refl["B04"]) / (refl["B08"] + refl["B04"])
                    nbr = (refl["B08"] - refl["B12"]) / (refl["B08"] + refl["B12"])
                    nbr[~clear] = np.nan
                    if nbr0 is None:
                        nbr0 = nbr.copy()
                    dnbr = nbr0 - nbr
                    ok = np.isfinite(dnbr)
                    done[key] = {"satelite": "Sentinel-2", "data": acq.day, "nuvem_area": round(c, 3), "cenas": acq.ids,
                                 "imagens": {"natural": f1, "queimada": f2},
                                 "ndvi_mediana": med(ndvi, clear), "nbr_mediana": med(nbr),
                                 "queimada_pct": round(float((dnbr[ok] > 0.27).mean() * 100), 1) if ok.any() else None,
                                 "agua_pct": round(float(water[clear].mean() * 100), 1) if clear.any() else None}
                r["s2"] = dict(done[key], dias=dd)
                log(f"  Sentinel-2 {acq.day} (±{dd} d, nuvem {c:.0%})")
            else:
                log("  Sentinel-2: nada limpo na janela")
        except Exception as e:
            log(f"  Sentinel-2 falhou: {type(e).__name__}: {e}")
        # Landsat 8/9
        try:
            pick = next(((dd, x, c) for dd, x in nearest(ls, day, WIN["landsat"]) for c in [ls_cloud(x)] if c <= MAX_CLOUD), None)
            if pick:
                dd, acq, c = pick
                key = "ls" + acq.day
                if key not in done:
                    sr, lst, clear, water = collect.landsat_load(acq, grid)
                    ds = _load(acq.items, ["green", "blue"], grid)
                    g = ds["green"].values[0].astype("float64"); b = ds["blue"].values[0].astype("float64")
                    g[g == 0] = np.nan; b[b == 0] = np.nan
                    g, b = g * 0.0000275 - 0.2, b * 0.0000275 - 0.2
                    nat = np.dstack([fixed(sr["red"], 0, 0.22, 0.8), fixed(g, 0, 0.20, 0.8), fixed(b, 0, 0.18, 0.8)])
                    tmp = plt.get_cmap("inferno")(fixed(lst, 20, 55))[..., :3]
                    tmp[~np.isfinite(lst)] = 0
                    plat = acq.items[0].properties.get("platform", "landsat").replace("landsat-", "Landsat ")
                    f1, f2 = f"landsat_{acq.day.replace('-', '')}_natural.png", f"landsat_{acq.day.replace('-', '')}_temperatura.png"
                    save(out / f1, nat); save(out / f2, tmp)
                    done[key] = {"satelite": plat, "data": acq.day, "nuvem_area": round(c, 3), "cenas": acq.ids,
                                 "imagens": {"natural": f1, "temperatura": f2}, "temperatura_mediana_c": med(lst, clear),
                                 "ndvi_mediana": med((sr["nir08"] - sr["red"]) / (sr["nir08"] + sr["red"]), clear)}
                r["landsat"] = dict(done[key], dias=dd)
                log(f"  {done[key]['satelite']} {acq.day} (±{dd} d, nuvem {c:.0%})")
        except Exception as e:
            log(f"  Landsat falhou: {type(e).__name__}: {e}")
        # Sentinel-1 RTC (radar banda C, sem nuvem)
        try:
            near = nearest(s1, day, WIN["s1"])
            if near:
                dd, acq = near[0]
                key = "s1" + acq.day + str(acq.key[1:])
                if key not in done:
                    v = collect.sentinel1_load(acq, grid)
                    vv, vh = 10 * np.log10(v["vv"]), 10 * np.log10(v["vh"])
                    rgb = np.dstack([fixed(vv, -20, 0), fixed(vh, -27, -7), fixed(vv - vh, 0, 12)])
                    f1 = f"s1_{acq.day.replace('-', '')}_vvvh.png"
                    save(out / f1, rgb)
                    done[key] = {"satelite": "Sentinel-1", "data": acq.day, "orbita": f"{acq.key[1]} {acq.key[2]}", "cenas": acq.ids,
                                 "imagens": {"radar": f1}, "vv_mediana_db": med(vv), "vh_mediana_db": med(vh)}
                r["s1"] = dict(done[key], dias=dd)
                log(f"  Sentinel-1 {acq.day} (±{dd} d)")
        except Exception as e:
            log(f"  Sentinel-1 falhou: {type(e).__name__}: {e}")
        rec.append(r)
        (out / "carrossel.json").write_text(json.dumps(report._clean({"bbox": list(grid.bbox), "crs": str(grid.crs), "res_m": a.res,
                                                                       "janelas_dias": WIN, "datas": rec}), ensure_ascii=False, indent=1), encoding="utf-8")
    n = {k: sum(1 for x in rec if k in x) for k in ("s2", "landsat", "s1")}
    log(f"pronto: {len(rec)} datas · Sentinel-2 {n['s2']} · Landsat {n['landsat']} · Sentinel-1 {n['s1']} · {out}")


if __name__ == "__main__":
    main()
