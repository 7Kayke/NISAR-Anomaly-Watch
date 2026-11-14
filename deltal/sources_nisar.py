"""NISAR: busca no catálogo da NASA (CMR, via earthaccess), leitura SÓ do recorte e reprojeção.

Login: variáveis de ambiente EARTHDATA_USERNAME e EARTHDATA_PASSWORD (Secrets no GitHub)
ou arquivo ~/.netrc.
"""
import re

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

NAME_RE = re.compile(r"NISAR_(L\d)_(\w{2})_([A-Z0-9]+)_(\d{3})_(\d{3})_([AD])_(\d{3})")
COLLECTIONS = {
    ("GCOV", "PROVISIONAL"): "NISAR_L2_GCOV_PROVISIONAL_V1",
    ("GUNW", "PROVISIONAL"): "NISAR_L2_GUNW_PROVISIONAL_V1",
}


def parse_name(name):
    m = NAME_RE.search(name)
    if not m:
        return {}
    times = re.findall(r"\d{8}T\d{6}", name)
    crid = re.search(r"_([PX]\d{5})_", name)
    return {"nivel": m.group(1), "colecao": m.group(2), "produto": m.group(3),
            "ciclo": int(m.group(4)), "pista": int(m.group(5)), "direcao": m.group(6),
            "quadro": int(m.group(7)), "inicio": times[0] if times else None,
            "crid": crid.group(1) if crid else None}


def login():
    import earthaccess
    auth = earthaccess.login(strategy="environment")
    if not auth or not getattr(auth, "authenticated", True):
        auth = earthaccess.login(strategy="netrc")
    return auth


def search(bbox, start, end, short_name="NISAR_L2_GCOV_PROVISIONAL_V1", granule_names=None):
    import earthaccess
    if granule_names:
        res = []
        for g in granule_names:
            res += earthaccess.search_data(short_name=short_name, granule_name=g)
        return res
    return earthaccess.search_data(short_name=short_name, bounding_box=tuple(bbox), temporal=(start, end))


def group_by_geometry(granules):
    groups = {}
    for g in granules:
        name = g["umm"]["GranuleUR"]
        info = parse_name(name)
        key = (info.get("pista"), info.get("direcao"), info.get("quadro"), info.get("crid", "")[:1])
        groups.setdefault(key, []).append((info.get("inicio"), name, g))
    for k in groups:
        groups[k].sort()
    return groups


def _find(h5, wanted):
    hits = {}
    def visit(name, obj):
        base = name.rsplit("/", 1)[-1]
        if base in wanted and "frequencyA" in name and base not in hits:
            hits[base] = name
    h5.visititems(visit)
    return hits


def _epsg(h5, group):
    for cand in [group + "/projection", group.rsplit("/", 1)[0] + "/projection"]:
        if cand in h5:
            ds = h5[cand]
            for k, v in ds.attrs.items():
                if "epsg" in k.lower():
                    return int(np.atleast_1d(v)[0])
            try:
                return int(ds[()])
            except Exception:
                pass
    raise KeyError("Não achei o código EPSG do GCOV; liste a árvore do arquivo e ajuste _epsg().")


def read_gcov_subset(granule, grid, layers=("HHHH", "HVHV")):
    """Lê só a janela da área de cada camada e reprojeta para a grade comum."""
    import earthaccess
    import h5py
    f = earthaccess.open([granule])[0]
    with h5py.File(f, "r") as h5:
        paths = _find(h5, set(layers) | {"xCoordinates", "yCoordinates"})
        missing = [l for l in layers if l not in paths]
        if missing:
            raise KeyError(f"Camadas ausentes {missing}; encontradas: {sorted(paths)}")
        group = paths[layers[0]].rsplit("/", 1)[0]
        x = h5[group + "/xCoordinates"][()] if group + "/xCoordinates" in h5 else h5[paths["xCoordinates"]][()]
        y = h5[group + "/yCoordinates"][()] if group + "/yCoordinates" in h5 else h5[paths["yCoordinates"]][()]
        epsg = _epsg(h5, group)
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        w, s, e, n = grid.bbox
        xs, ys = zip(*[tr.transform(a, b) for a, b in [(w, s), (w, n), (e, s), (e, n)]])
        pad = 3 * abs(x[1] - x[0])
        ix = np.where((x >= min(xs) - pad) & (x <= max(xs) + pad))[0]
        iy = np.where((y >= min(ys) - pad) & (y <= max(ys) + pad))[0]
        if ix.size == 0 or iy.size == 0:
            raise ValueError("A área não cai dentro desta cena")
        x0, x1, y0, y1 = ix.min(), ix.max() + 1, iy.min(), iy.max() + 1
        dx, dy = x[1] - x[0], y[1] - y[0]
        src_tr = Affine(dx, 0, x[x0] - dx / 2, 0, dy, y[y0] - dy / 2)
        out = {}
        for lay in layers:
            arr = h5[paths[lay]][y0:y1, x0:x1].astype("float32")
            arr[~np.isfinite(arr) | (arr <= 0)] = np.nan
            dst = np.full((grid.height, grid.width), np.nan, dtype="float32")
            reproject(arr, dst, src_transform=src_tr, src_crs=f"EPSG:{epsg}", src_nodata=np.nan,
                      dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                      resampling=Resampling.average)
            out[lay] = dst.astype("float64")
    meta = parse_name(granule["umm"]["GranuleUR"])
    meta.update({"arquivo": granule["umm"]["GranuleUR"], "epsg_produto": epsg,
                 "janela_linhas": [int(y0), int(y1)], "janela_colunas": [int(x0), int(x1)],
                 "grupo_hdf5": group})
    return out, meta
