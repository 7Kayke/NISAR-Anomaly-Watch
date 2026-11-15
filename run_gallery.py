# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Galeria de comparação: Landsat 8/9 (NASA/USGS) e Sentinel-2 (ESA) no MESMO recorte da área,
nas janelas de data da história, com o contorno do que o NISAR detectou por cima.

Uso:
    python run_gallery.py --case cases/paiaguas.json --serie docs/results/paiaguas --out docs/results/paiaguas/galeria

Para cada janela (antes / depois / início) escolhe a cena com menos nuvem DENTRO da área e gera:
  landsat_<janela>_cor_natural.png    vermelho, verde, azul
  landsat_<janela>_queimada.png       SWIR2, NIR, vermelho: cicatriz de fogo em marrom-avermelhado, vegetação em verde
  landsat_<janela>_temperatura.png    temperatura da superfície (°C), banda térmica
  sentinel2_<janela>_cor_natural.png / _queimada.png
  galeria.json                        data, satélite, cena e nuvem de cada imagem
O contorno amarelo é a queda abrupta do NISAR (classe 3) e o tracejado ciano a perda de coerência (< 0,3).
São imagens de CONTEXTO: o projeto continua sendo sobre o NISAR.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deltal import collect, report
from deltal.grid import make_grid
from deltal.sources_pc import _search, _load, S2_CLOUD_SCL, _s2_offset

WINDOWS = [("inicio", "2026-06-15", "2026-07-15"), ("antes", "2026-08-10", "2026-09-08"), ("depois", "2026-09-20", "2026-10-25")]


def stretch(rgb, lo=2, hi=98):
    out = np.zeros_like(rgb)
    for i in range(3):
        b = rgb[..., i]
        v = b[np.isfinite(b)]
        if v.size:
            p1, p2 = np.percentile(v, [lo, hi])
            out[..., i] = np.clip((b - p1) / (p2 - p1 + 1e-9), 0, 1)
    return np.nan_to_num(out)


def masks_from_series(serie, grid):
    """Queda abrupta (classe 3) da geometria principal e coerência média, na grade da galeria."""
    from rasterio.warp import reproject, Resampling
    import rasterio
    drop = coh = None
    serie = Path(serie) if serie else None
    if serie and serie.exists():
        geos = sorted([p for p in serie.iterdir() if (p / "dados" / "mapas.tif").exists()],
                      key=lambda p: -len(json.loads((p / "app" / "app.json").read_text())["datas"]) if (p / "app" / "app.json").exists() else 0)
        if geos:
            with rasterio.open(geos[0] / "dados" / "mapas.tif") as r:
                dst = np.zeros((grid.height, grid.width), "float32")
                reproject(r.read(1), dst, src_transform=r.transform, src_crs=r.crs, dst_transform=grid.transform,
                          dst_crs=grid.crs, resampling=Resampling.nearest)
                drop = dst == 3
        tif = serie / "gunw" / "coerencia_media.tif"
        if tif.exists():
            with rasterio.open(tif) as r:
                dst = np.full((grid.height, grid.width), np.nan, "float32")
                reproject(r.read(1), dst, src_transform=r.transform, src_crs=r.crs, dst_transform=grid.transform,
                          dst_crs=grid.crs, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
                coh = dst < 0.3
    return drop, coh


def figure(path, img, title, sub, drop, coh, cmap=None, vmin=None, vmax=None, cbar=None, point=None):
    fig, ax = plt.subplots(figsize=(7, 7.4), dpi=120)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if drop is not None and drop.any():
        ax.contour(drop.astype(float), levels=[0.5], colors=["#EAFE07"], linewidths=1.3)
    if coh is not None and coh.any():
        ax.contour(coh.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=0.9, linestyles="--")
    if point is not None:
        ax.plot(*point, marker="o", ms=11, mfc="none", mec="#FF6B00", mew=2.2)
    ax.set_axis_off()
    ax.set_title(title, color=report.DEEP, fontsize=12, loc="left")
    fig.text(0.02, 0.015, sub, fontsize=8.5, color="#44506A")
    if cbar:
        c = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); c.set_label(cbar)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def best_scene(collection, grid, t0, t1, cloud_fn, query=None):
    items = _search(collection, grid.bbox, t0, t1, query)
    if collection == "landsat-c2-l2":
        items = [i for i in items if i.properties.get("platform") in ("landsat-8", "landsat-9")]
    acqs = collect._cover(collect._acquisitions(items), grid.bbox)
    best = None
    for a in acqs:
        c = cloud_fn(a)
        if c <= 0.3 and (best is None or c < best[1] or (c == best[1] and "landsat-9" in str(a.items[0].properties.get("platform")))):
            best = (a, c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--serie"); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=20)
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(case["bbox"], a.res)
    windows = case.get("galeria", WINDOWS)
    drop, coh = masks_from_series(a.serie, grid)
    pt = None
    if case.get("ponto"):
        from pyproj import Transformer
        x, y = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(case["ponto"]["lon"], case["ponto"]["lat"])
        pt = ((x - grid.transform.c) / grid.transform.a, (y - grid.transform.f) / grid.transform.e)
    qg = make_grid(grid.bbox, 100)
    def ls_cloud(acq):
        qa = _load(acq.items, ["qa_pixel"], qg, resampling="nearest")["qa_pixel"].values[0].astype("int64")
        cl = np.zeros(qa.shape, bool)
        for b in collect.LANDSAT_CLOUD_BITS:
            cl |= ((qa >> b) & 1) == 1
        return float(cl.mean())
    def s2_cloud(acq):
        scl = _load(acq.items, ["SCL"], qg, resampling="nearest")["SCL"].values[0]
        return float(np.isin(scl, list(S2_CLOUD_SCL)).mean())
    leg = "Contorno amarelo: queda abrupta no NISAR (HV) · tracejado ciano: perda de coerência NISAR (< 0,3) · círculo: o ponto"
    rec = []
    for name, t0, t1 in windows:
        print(f"[{name}] {t0} a {t1}", flush=True)
        try:
            b = best_scene("landsat-c2-l2", grid, t0, t1, ls_cloud, {"eo:cloud_cover": {"lt": 80}})
            if b:
                acq, c = b
                sr, lst, clear, water = collect.landsat_load(acq, grid)
                ds = _load(acq.items, ["green", "blue"], grid)
                g = ds["green"].values[0].astype("float64"); bl = ds["blue"].values[0].astype("float64")
                g[g == 0] = np.nan; bl[bl == 0] = np.nan
                g, bl = g * 0.0000275 - 0.2, bl * 0.0000275 - 0.2
                plat = acq.items[0].properties.get("platform", "landsat").replace("landsat-", "Landsat ")
                day = acq.day
                figure(out / f"landsat_{name}_cor_natural.png", stretch(np.dstack([sr["red"], g, bl])), f"{plat} · cor natural · {day}",
                       f"NASA/USGS Landsat C2 L2 · nuvem na área {c:.0%} · {leg}", drop, coh, point=pt)
                figure(out / f"landsat_{name}_queimada.png", stretch(np.dstack([sr["swir22"], sr["nir08"], sr["red"]])),
                       f"{plat} · falsa cor (SWIR2, NIR, vermelho) · {day}", f"Cicatriz de fogo em marrom-avermelhado, vegetação em verde · {leg}", drop, coh, point=pt)
                v = lst[np.isfinite(lst)]
                lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (20, 45)
                figure(out / f"landsat_{name}_temperatura.png", lst, f"{plat} · temperatura da superfície · {day}",
                       f"Banda térmica (ST_B10) em °C · {leg}", drop, coh, cmap="inferno", vmin=lo, vmax=hi, cbar="°C", point=pt)
                rec.append({"janela": name, "satelite": plat, "data": day, "nuvem_area": round(c, 3), "cenas": acq.ids,
                            "imagens": [f"landsat_{name}_cor_natural.png", f"landsat_{name}_queimada.png", f"landsat_{name}_temperatura.png"],
                            "temperatura_mediana_c": round(float(np.nanmedian(lst)), 1) if v.size else None,
                            "temperatura_na_queda_c": round(float(np.nanmedian(lst[drop & np.isfinite(lst)])), 1) if drop is not None and (drop & np.isfinite(lst)).any() else None})
                print(f"  Landsat {day} nuvem {c:.0%}", flush=True)
            else:
                print("  Landsat: nenhuma cena com pouca nuvem", flush=True)
        except Exception as e:
            print(f"  Landsat falhou: {type(e).__name__}: {e}", flush=True)
        try:
            b = best_scene("sentinel-2-l2a", grid, t0, t1, s2_cloud, {"eo:cloud_cover": {"lt": 80}})
            if b:
                acq, c = b
                refl, clear, water = collect.sentinel2_load(acq, grid)
                day = acq.day
                figure(out / f"sentinel2_{name}_cor_natural.png", stretch(np.dstack([refl["B04"], refl["B03"], refl["B02"]])),
                       f"Sentinel-2 · cor natural · {day}", f"ESA Copernicus Sentinel-2 L2A · nuvem na área {c:.0%} · {leg}", drop, coh, point=pt)
                figure(out / f"sentinel2_{name}_queimada.png", stretch(np.dstack([refl["B12"], refl["B08"], refl["B04"]])),
                       f"Sentinel-2 · falsa cor (SWIR, NIR, vermelho) · {day}", f"Cicatriz de fogo em marrom-avermelhado · {leg}", drop, coh, point=pt)
                rec.append({"janela": name, "satelite": "Sentinel-2", "data": day, "nuvem_area": round(c, 3), "cenas": acq.ids,
                            "imagens": [f"sentinel2_{name}_cor_natural.png", f"sentinel2_{name}_queimada.png"]})
                print(f"  Sentinel-2 {day} nuvem {c:.0%}", flush=True)
        except Exception as e:
            print(f"  Sentinel-2 falhou: {type(e).__name__}: {e}", flush=True)
    (out / "galeria.json").write_text(json.dumps(report._clean(rec), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
