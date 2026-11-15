# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Microsoft Planetary Computer: Sentinel-2 L2A, Sentinel-1 RTC e Copernicus DEM.

O catálogo STAC é público; as URLs dos arquivos são assinadas pelo pacote
planetary_computer (sem conta). Tudo é carregado já na grade comum (Grid).
"""
from datetime import datetime, timedelta

import numpy as np
import planetary_computer
import pystac_client
import odc.stac
from odc.geo.geobox import GeoBox

# leitura em nuvem mais rápida: não lista diretórios, reaproveita conexões HTTP
odc.stac.configure_rio(cloud_defaults=True, GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MAX_RETRY="4",
                       GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_CLOUD_SCL = {0, 1, 3, 8, 9, 10}   # sem dado, saturado, sombra, nuvem média/alta, cirrus
S2_WATER_SCL = {6}


def _client():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _geobox(grid):
    return GeoBox((grid.height, grid.width), grid.transform, grid.crs)


def _search(collection, bbox, start, end, query=None):
    items = _client().search(collections=[collection], bbox=list(bbox),
                             datetime=f"{start}/{end}", query=query or {}).item_collection()
    return sorted(items, key=lambda it: it.datetime)


def _load(items, bands, grid, resampling="bilinear"):
    return odc.stac.load(items, bands=bands, geobox=_geobox(grid), groupby="solar_day",
                         resampling=resampling, chunks=None, pool=8)


def _day(dt):
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- Sentinel-2
def _s2_offset(items):
    try:
        base = float(items[0].properties.get("s2:processing_baseline", "0"))
    except ValueError:
        base = 0.0
    return -1000.0 if base >= 4.0 else 0.0


def s2_pair(grid, event_date, window_days=30, max_cloud=0.2):
    """Escolhe a melhor cena antes e depois do evento (menos nuvem DENTRO da área)."""
    ev = datetime.fromisoformat(event_date)
    start = _day(ev - timedelta(days=window_days))
    end = _day(ev + timedelta(days=window_days))
    items = _search("sentinel-2-l2a", grid.bbox, start, end, {"eo:cloud_cover": {"lt": 60}})
    by_day = {}
    for it in items:
        by_day.setdefault(_day(it.datetime), []).append(it)
    candidates = []
    for day, its in by_day.items():
        scl = _load(its, ["SCL"], grid, resampling="nearest")["SCL"].values[0]
        cloud = float(np.isin(scl, list(S2_CLOUD_SCL)).mean())
        candidates.append((day, cloud, its))
    pre = [c for c in candidates if c[0] < event_date and c[1] <= max_cloud]
    post = [c for c in candidates if c[0] > event_date and c[1] <= max_cloud]
    if not pre or not post:
        raise RuntimeError(f"Sentinel-2 sem par limpo (antes={len(pre)}, depois={len(post)}); "
                           f"candidatos: {[(c[0], round(c[1], 2)) for c in candidates]}")
    return max(pre, key=lambda c: c[0]), min(post, key=lambda c: c[0])


def s2_load(day_items, grid):
    day, cloud, items = day_items
    ds = _load(items, ["B03", "B04", "B08", "B12", "SCL"], grid)
    off = _s2_offset(items)
    refl = {b: (ds[b].values[0].astype("float64") + off) / 10000.0 for b in ["B03", "B04", "B08", "B12"]}
    scl = ds["SCL"].values[0]
    clear = ~np.isin(scl, list(S2_CLOUD_SCL))
    water = np.isin(scl, list(S2_WATER_SCL))
    meta = {"data": day, "nuvem_na_area": round(cloud, 3), "offset": off,
            "itens": [it.id for it in items],
            "baseline": items[0].properties.get("s2:processing_baseline")}
    return refl, clear, water, meta


# ---------------------------------------------------------------- Sentinel-1 RTC
def s1_pair(grid, event_date, window_days=36):
    """Par antes/depois na MESMA órbita relativa e direção, o mais próximo do evento."""
    ev = datetime.fromisoformat(event_date)
    items = _search("sentinel-1-rtc", grid.bbox, _day(ev - timedelta(days=window_days)),
                    _day(ev + timedelta(days=window_days)))
    groups = {}
    for it in items:
        key = (it.properties.get("sat:relative_orbit"), it.properties.get("sat:orbit_state"))
        groups.setdefault(key, []).append(it)
    best = None
    for key, its in groups.items():
        pre = [i for i in its if _day(i.datetime) < event_date]
        post = [i for i in its if _day(i.datetime) > event_date]
        if pre and post:
            a, b = max(pre, key=lambda i: i.datetime), min(post, key=lambda i: i.datetime)
            gap = (b.datetime - a.datetime).days
            if best is None or gap < best[0]:
                best = (gap, key, a, b)
    if best is None:
        raise RuntimeError("Sentinel-1 RTC: nenhum par na mesma órbita atravessa o evento")
    return best


def s1_load(item, grid):
    ds = _load([item], ["vv", "vh"], grid)
    out = {}
    for b in ["vv", "vh"]:
        v = ds[b].values[0].astype("float64")
        v[~np.isfinite(v) | (v <= 0)] = np.nan
        out[b] = v
    meta = {"id": item.id, "data": item.datetime.isoformat(),
            "orbita_relativa": item.properties.get("sat:relative_orbit"),
            "direcao": item.properties.get("sat:orbit_state"),
            "plataforma": item.properties.get("platform")}
    return out, meta


# ---------------------------------------------------------------- DEM
def dem_load(grid):
    items = _search("cop-dem-glo-30", grid.bbox, "2000-01-01", "2100-01-01")
    ds = _load(items, ["data"], grid)
    return ds["data"].values[0].astype("float64"), [it.id for it in items]
