"""Coleta unificada no Microsoft Planetary Computer, passando pelo funil de cada fonte.

Fontes: sentinel-2-l2a, landsat-c2-l2 (óptico + térmico), sentinel-1-rtc, naip, aster-l1t,
cop-dem-glo-30. O NISAR passa por um funil equivalente em sources_nisar (catálogo da NASA).
A unidade do funil é a AQUISIÇÃO: itens do mesmo dia (e da mesma órbita, no radar) são unidos,
porque a área pode cair na divisa de dois tiles.
"""
from datetime import datetime, timedelta

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .grid import make_grid
from .funnel import Funnel, pair_by_key, MIN_COVER
from .sources_pc import _search, _load, S2_CLOUD_SCL, S2_WATER_SCL, _s2_offset

LANDSAT_CLOUD_BITS = (1, 2, 3, 4)       # nuvem dilatada, cirro, nuvem, sombra de nuvem
LANDSAT_WATER_BIT = 7
MAX_CLOUD = 0.2
QA_RES_M = 100          # a checagem de nuvem roda numa grade grossa: 25x menos leitura, mesma fração


def _log(msg):
    print(f"    {msg}", flush=True)


class Acq:
    """Uma aquisição: itens do mesmo dia (+ mesma órbita) unidos."""
    def __init__(self, key, items):
        self.key = key
        self.items = sorted(items, key=lambda i: i.id)
        self.dt = min(i.datetime for i in items)
        self.geometry = unary_union([shape(i.geometry) for i in items])
        self.quality = {}

    @property
    def day(self):
        return self.dt.strftime("%Y-%m-%d")

    @property
    def ids(self):
        return [i.id for i in self.items]


def _acquisitions(items, extra_key=lambda it: ()):
    groups = {}
    for it in items:
        k = (it.datetime.strftime("%Y-%m-%d"),) + tuple(extra_key(it))
        groups.setdefault(k, []).append(it)
    return [Acq(k, its) for k, its in groups.items()]


def _cover(acqs, bbox):
    aoi = box(*bbox)
    out = []
    for a in acqs:
        a.quality["cobertura"] = round(a.geometry.intersection(aoi).area / aoi.area, 3)
        if a.quality["cobertura"] >= MIN_COVER:
            out.append(a)
    return out


def _window(event_date, days):
    ev = datetime.fromisoformat(event_date)
    return ev, (ev - timedelta(days=days)).strftime("%Y-%m-%d"), (ev + timedelta(days=days)).strftime("%Y-%m-%d")


def _pair_optical(fun, good, ev):
    best = pair_by_key(good, ev, key=lambda a: "optico", date_of=lambda a: a.dt.replace(tzinfo=None))
    return fun.add("pareados", [best[1], best[2]] if best else [],
                   "antes mais recente e depois mais cedo, com pouca nuvem"), best


# ------------------------------------------------------------------ Sentinel-2
def sentinel2(grid, event_date, days=30):
    ev, t0, t1 = _window(event_date, days)
    fun = Funnel("Sentinel-2 L2A", "sentinel-2-l2a")
    items = _search("sentinel-2-l2a", grid.bbox, t0, t1, {"eo:cloud_cover": {"lt": 80}})
    acqs = fun.add("catalogo", _acquisitions(items), f"{t0} a {t1}, nuvem na cena < 80%")
    acqs = fun.add("cobre_area", _cover(acqs, grid.bbox), f"cobre >= {int(MIN_COVER*100)}% da área")
    good = []
    qg = make_grid(grid.bbox, QA_RES_M)
    for n, a in enumerate(acqs, 1):
        _log(f"Sentinel-2 nuvem {n}/{len(acqs)} ({a.day})")
        scl = _load(a.items, ["SCL"], qg, resampling="nearest")["SCL"].values[0]
        a.quality["nuvem_na_area"] = round(float(np.isin(scl, list(S2_CLOUD_SCL)).mean()), 3)
        if a.quality["nuvem_na_area"] <= MAX_CLOUD:
            good.append(a)
    good = fun.add("qualidade", good, f"nuvem dentro da área <= {int(MAX_CLOUD*100)}% (camada SCL)")
    _, best = _pair_optical(fun, good, ev)
    return fun, best


def sentinel2_load(acq, grid):
    ds = _load(acq.items, ["B02", "B03", "B04", "B08", "B12"], grid)
    off = _s2_offset(acq.items)
    refl = {}
    for b in ["B02", "B03", "B04", "B08", "B12"]:
        v = ds[b].values[0].astype("float64")
        v[v == 0] = np.nan                     # 0 = sem dado
        refl[b] = (v + off) / 10000.0
    scl = _load(acq.items, ["SCL"], grid, resampling="nearest")["SCL"].values[0]   # classes: nunca interpolar
    return refl, ~np.isin(scl, list(S2_CLOUD_SCL)), np.isin(scl, list(S2_WATER_SCL))


# ------------------------------------------------------------------ Landsat C2 L2
def landsat(grid, event_date, days=40):
    ev, t0, t1 = _window(event_date, days)
    fun = Funnel("Landsat 8/9 Coleção 2 Nível 2 (óptico + térmico)", "landsat-c2-l2")
    items = _search("landsat-c2-l2", grid.bbox, t0, t1, {"eo:cloud_cover": {"lt": 80}})
    items = [i for i in items if i.properties.get("platform") in ("landsat-8", "landsat-9")]
    acqs = fun.add("catalogo", _acquisitions(items), f"{t0} a {t1}, Landsat 8 e 9, nuvem na cena < 80%")
    acqs = fun.add("cobre_area", _cover(acqs, grid.bbox), f"cobre >= {int(MIN_COVER*100)}% da área")
    good = []
    qg = make_grid(grid.bbox, QA_RES_M)
    for n, a in enumerate(acqs, 1):
        if not all("lwir11" in i.assets for i in a.items):
            continue
        _log(f"Landsat nuvem {n}/{len(acqs)} ({a.day})")
        qa = _load(a.items, ["qa_pixel"], qg, resampling="nearest")["qa_pixel"].values[0].astype("int64")
        cloud = np.zeros(qa.shape, bool)
        for b in LANDSAT_CLOUD_BITS:
            cloud |= (qa >> b) & 1 == 1
        a.quality["nuvem_na_area"] = round(float(cloud.mean()), 3)
        if a.quality["nuvem_na_area"] <= MAX_CLOUD:
            good.append(a)
    good = fun.add("qualidade", good, "banda térmica presente e nuvem dentro da área <= 20% (QA_PIXEL)")
    _, best = _pair_optical(fun, good, ev)
    return fun, best


def landsat_load(acq, grid):
    ds = _load(acq.items, ["red", "nir08", "swir22", "lwir11"], grid)
    sr = {}
    for b in ["red", "nir08", "swir22"]:
        v = ds[b].values[0].astype("float64")
        v[v == 0] = np.nan
        sr[b] = v * 0.0000275 - 0.2
    st = ds["lwir11"].values[0].astype("float64")
    st[st == 0] = np.nan
    lst_c = st * 0.00341802 + 149.0 - 273.15
    qa = _load(acq.items, ["qa_pixel"], grid, resampling="nearest")["qa_pixel"].values[0].astype("int64")
    clear = np.ones(qa.shape, bool)
    for b in LANDSAT_CLOUD_BITS:
        clear &= ((qa >> b) & 1) == 0
    water = ((qa >> LANDSAT_WATER_BIT) & 1) == 1
    return sr, lst_c, clear, water


# ------------------------------------------------------------------ Sentinel-1 RTC
def sentinel1(grid, event_date, days=36):
    ev, t0, t1 = _window(event_date, days)
    fun = Funnel("Sentinel-1 RTC (banda C)", "sentinel-1-rtc")
    items = _search("sentinel-1-rtc", grid.bbox, t0, t1)
    okey = lambda it: (it.properties.get("sat:relative_orbit"), it.properties.get("sat:orbit_state"))
    acqs = fun.add("catalogo", _acquisitions(items, okey), f"{t0} a {t1}")
    acqs = fun.add("cobre_area", _cover(acqs, grid.bbox), f"cobre >= {int(MIN_COVER*100)}% da área")
    good = fun.add("qualidade", [a for a in acqs if all({"vv", "vh"} <= set(i.assets) for i in a.items)],
                   "tem VV e VH")
    best = pair_by_key(good, ev, key=lambda a: a.key[1:], date_of=lambda a: a.dt.replace(tzinfo=None))
    fun.add("pareados", [best[1], best[2]] if best else [], "mesma órbita relativa e direção, menor intervalo")
    return fun, best


def sentinel1_load(acq, grid):
    ds = _load(acq.items, ["vv", "vh"], grid)
    out = {}
    for b in ["vv", "vh"]:
        v = ds[b].values[0].astype("float64")
        v[~np.isfinite(v) | (v <= 0)] = np.nan
        out[b] = v
    return out


# ------------------------------------------------------------------ NAIP (contexto "antes" em alta resolução)
def naip(grid, event_date, years=6):
    ev = datetime.fromisoformat(event_date)
    fun = Funnel("NAIP (foto aérea dos EUA, ~0,6 m)", "naip")
    items = _search("naip", grid.bbox, (ev - timedelta(days=365 * years)).strftime("%Y-%m-%d"), event_date)
    acqs = fun.add("catalogo", _acquisitions(items), f"{years} anos antes do evento")
    years_acq = {}
    for a in acqs:
        years_acq.setdefault(a.dt.year, []).append(a)
    merged = []
    for y, al in years_acq.items():
        m = Acq((str(y),), [i for a in al for i in a.items])
        merged.append(m)
    merged = fun.add("cobre_area", _cover(merged, grid.bbox), "mosaico do mesmo ano cobre a área")
    latest = sorted(merged, key=lambda a: a.dt)[-1:] if merged else []
    fun.add("pareados", latest, "só existe 'antes': NAIP é refeito a cada 2-3 anos")
    return fun, (latest[0] if latest else None)


def naip_load_rgb(acq, grid, resolution=4.0):
    """Mosaico RGB do NAIP reamostrado para a área (o asset 'image' tem 4 bandas: R, G, B, NIR)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from .grid import make_grid
    g = make_grid(grid.bbox, resolution)
    rgb = np.zeros((3, g.height, g.width), "uint8")
    for it in acq.items:
        with rasterio.open(it.assets["image"].href) as src:
            for b in range(3):
                dst = np.zeros((g.height, g.width), "uint8")
                reproject(rasterio.band(src, b + 1), dst, dst_transform=g.transform, dst_crs=g.crs,
                          resampling=Resampling.average, dst_nodata=0)
                rgb[b] = np.where(dst > 0, dst, rgb[b])
    return np.moveaxis(rgb, 0, -1), g


# ------------------------------------------------------------------ ASTER L1T e DEM
def aster(grid, event_date, days=60):
    ev, t0, t1 = _window(event_date, days)
    fun = Funnel("ASTER L1T (térmico 90 m)", "aster-l1t")
    items = _search("aster-l1t", grid.bbox, t0, t1)
    acqs = fun.add("catalogo", _acquisitions(items), f"{t0} a {t1}")
    acqs = fun.add("cobre_area", _cover(acqs, grid.bbox), f"cobre >= {int(MIN_COVER*100)}% da área")
    best = pair_by_key(acqs, ev, key=lambda a: "aster", date_of=lambda a: a.dt.replace(tzinfo=None))
    fun.add("pareados", [best[1], best[2]] if best else [], "antes e depois")
    return fun, best


def dem(grid):
    fun = Funnel("Copernicus DEM 30 m", "cop-dem-glo-30")
    items = _search("cop-dem-glo-30", grid.bbox, "2000-01-01", "2100-01-01")
    acq = Acq(("dem",), items) if items else None
    fun.add("catalogo", [acq] if acq else [], "modelo de relevo")
    fun.add("cobre_area", _cover([acq], grid.bbox) if acq else [], "cobre a área")
    if not acq:
        return fun, None
    z = _load(acq.items, ["data"], grid)["data"].values[0].astype("float64")
    return fun, z
