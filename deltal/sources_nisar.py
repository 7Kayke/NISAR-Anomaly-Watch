# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
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
    mode = re.search(r"_(\d{4})_([A-Z]{4})_", name)
    out = {"nivel": m.group(1), "colecao": m.group(2), "produto": m.group(3),
           "ciclo": int(m.group(4)), "pista": int(m.group(5)), "direcao": m.group(6),
           "quadro": int(m.group(7)), "inicio": times[0] if times else None,
           "crid": crid.group(1) if crid else None}
    if mode:
        out["modo_banda"], out["polarizacao"] = mode.group(1), mode.group(2)
    if len(times) >= 3 and out["produto"] in ("GUNW", "RIFG", "RUNW", "GOFF", "ROFF"):
        out["inicio_par2"] = times[2]          # produtos de par: referência e segunda data
    return out


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


def _umm_geometry(g):
    try:
        pts = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        return {"type": "Polygon", "coordinates": [[(p["Longitude"], p["Latitude"]) for p in pts]]}
    except (KeyError, IndexError, TypeError):
        return None


def funnel(grid, event_date, days=36,
           short_names=("NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1")):
    """Funil do NISAR: catálogo -> cobre a área -> mesma maturidade -> par na mesma geometria."""
    from datetime import datetime, timedelta
    from .funnel import Funnel, coverage, pair_by_key, MIN_COVER
    ev = datetime.fromisoformat(event_date)
    t0 = (ev - timedelta(days=days)).strftime("%Y-%m-%d")
    t1 = (ev + timedelta(days=days)).strftime("%Y-%m-%d")
    fun = Funnel("NISAR GCOV (banda L)", " + ".join(short_names))
    grans = []
    for sn in short_names:
        for g in search(grid.bbox, t0, t1, short_name=sn):
            grans.append((sn, g))
    cat = fun.add("catalogo", grans, f"{t0} a {t1}, coleções {', '.join(short_names)}")
    cov = []
    for sn, g in cat:
        geom = _umm_geometry(g)
        if geom and coverage(geom, grid.bbox) >= MIN_COVER:
            cov.append((sn, g))
    cov = fun.add("cobre_area", cov, f"cobre >= {int(MIN_COVER*100)}% da área")
    def dt(x):
        return datetime.strptime(parse_name(x[1]["umm"]["GranuleUR"])["inicio"], "%Y%m%dT%H%M%S")
    def geo(x):
        i = parse_name(x[1]["umm"]["GranuleUR"])
        return (i["pista"], i["direcao"], i["quadro"])
    best = None
    for maturity in short_names:          # prefere um par inteiro dentro da MESMA coleção
        same = [x for x in cov if x[0] == maturity]
        best = pair_by_key(same, ev, key=geo, date_of=dt)
        if best:
            fun.add("qualidade", same, f"mesma coleção ({maturity})")
            break
    if not best:
        best = pair_by_key(cov, ev, key=geo, date_of=dt)
        fun.add("qualidade", cov, "coleções misturadas (declarar na ficha)")
    fun.add("pareados", [best[1], best[2]] if best else [], "mesma pista, direção e quadro; menor intervalo")
    return fun, best


def _umm_meta(sn, g):
    """Metadados do catálogo (CMR/UMM) de um grânulo, achatados para a ficha."""
    u = g["umm"]
    name = u.get("GranuleUR", "")
    meta = {"arquivo": name, "colecao_cmr": sn}
    meta.update(parse_name(name))
    te = u.get("TemporalExtent", {}).get("RangeDateTime", {})
    meta["inicio_utc"], meta["fim_utc"] = te.get("BeginningDateTime"), te.get("EndingDateTime")
    try:
        meta["tamanho_mb"] = round(sum(f.get("Size", 0) * (1024 if f.get("SizeUnit") == "GB" else 1)
                                       for f in u["DataGranule"]["ArchiveAndDistributionInformation"]), 1)
    except (KeyError, TypeError):
        pass
    for a in u.get("AdditionalAttributes", []) or []:
        v = a.get("Values", [])
        meta["attr_" + a.get("Name", "?")] = v[0] if len(v) == 1 else v
    meta["produzido_em"] = (u.get("DataGranule", {}) or {}).get("ProductionDateTime")
    return meta


MATURITY = {"NISAR_L2_GCOV_V1": 0, "NISAR_L2_GCOV_PROVISIONAL_V1": 1}


def funnel_series(grid, start, end, short_names=("NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1"),
                  geometry=None, min_cover=None):
    """Funil da SÉRIE: catálogo -> cobre a área -> uma versão por passagem -> mesma geometria.

    Devolve (Funnel, escolhidos [(short_name, granulo)] em ordem de data, metadados de TODOS os
    grânulos do catálogo com a etapa em que cada um parou e o porquê)."""
    from .funnel import Funnel, coverage, MIN_COVER
    MIN_COVER = MIN_COVER if min_cover is None else min_cover
    fun = Funnel("NISAR GCOV (banda L), série temporal", " + ".join(short_names))
    grans = [(sn, g) for sn in short_names for g in search(grid.bbox, start, end, short_name=sn)]
    meta = {}
    for sn, g in grans:
        m = _umm_meta(sn, g)
        geom = _umm_geometry(g)
        m["cobertura_area"] = round(coverage(geom, grid.bbox), 3) if geom else None
        m["etapa_final"], m["motivo"] = "catalogo", ""
        meta[m["arquivo"]] = m
    fun.add("catalogo", grans, f"{start} a {end}, coleções {', '.join(short_names)}")
    cov = [(sn, g) for sn, g in grans if (meta[g["umm"]["GranuleUR"]]["cobertura_area"] or 0) >= MIN_COVER]
    for sn, g in grans:
        mm = meta[g["umm"]["GranuleUR"]]
        if (sn, g) in cov:
            mm["etapa_final"] = "cobre_area"
        else:
            mm["motivo"] = f"cobre só {mm['cobertura_area']} da área"
    fun.add("cobre_area", cov, f"cobre >= {int(MIN_COVER * 100)}% da área")
    # uma versão por passagem (ciclo, pista, direção, quadro): prefere a validada
    best = {}
    for sn, g in cov:
        i = parse_name(g["umm"]["GranuleUR"])
        key = (i["ciclo"], i["pista"], i["direcao"], i["quadro"])
        cur = best.get(key)
        if cur is None or (MATURITY.get(sn, 9), g["umm"]["GranuleUR"]) < (MATURITY.get(cur[0], 9), cur[1]["umm"]["GranuleUR"]):
            best[key] = (sn, g)
    uniq = list(best.values())
    for sn, g in cov:
        mm = meta[g["umm"]["GranuleUR"]]
        if (sn, g) in uniq:
            mm["etapa_final"] = "uma_por_passagem"
        else:
            mm["motivo"] = "mesma passagem já existe em versão preferida (validada > provisória)"
    fun.add("uma_por_passagem", uniq, "duplicatas da mesma passagem: fica a validada")
    groups = {}
    for sn, g in uniq:
        i = parse_name(g["umm"]["GranuleUR"])
        groups.setdefault((i["pista"], i["direcao"], i["quadro"]), []).append((sn, g))
    chosen, gkey = [], None
    if geometry:
        gkey = tuple(geometry)
        groups.setdefault(gkey, [])
    elif groups:
        gkey = max(groups, key=lambda k: (len(groups[k]), -k[0]))
    if gkey is not None:
        chosen = sorted(groups[gkey], key=lambda x: parse_name(x[1]["umm"]["GranuleUR"])["inicio"])
    for sn, g in uniq:
        mm = meta[g["umm"]["GranuleUR"]]
        if (sn, g) in chosen:
            mm["etapa_final"] = "mesma_geometria"
        else:
            mm["motivo"] = f"outra geometria; a escolhida (pista, direção, quadro) = {gkey}, com mais passagens"
    fun.add("mesma_geometria", chosen, f"geometria com mais passagens: pista/direção/quadro {gkey}")
    others = {f"{k[0]:03d}_{k[1]}_{k[2]:03d}": len(v) for k, v in groups.items()}
    return fun, chosen, sorted(meta.values(), key=lambda m: (m.get("inicio") or "", m["arquivo"])), others


def _find(h5, wanted):
    hits = {}
    def visit(name, obj):
        base = name.rsplit("/", 1)[-1]
        if base in wanted and "frequencyA" in name and base not in hits:
            hits[base] = name
    h5.visititems(visit)
    return hits


def _locate(h5, base, contains=None):
    """Caminho do dataset `base` na frequencyA, de preferência dentro de um grupo que contenha `contains`."""
    cands = []
    def visit(name, obj):
        if name.rsplit("/", 1)[-1] == base and "frequencyA" in name and hasattr(obj, "shape") and len(obj.shape) == 2:
            cands.append(name)
    h5.visititems(visit)
    if contains:
        pref = [c for c in cands if contains in c]
        cands = pref or cands
    if not cands:
        raise KeyError(f"{base} não encontrado; datasets 2D na frequencyA: {_tree(h5)[:40]}")
    return sorted(cands, key=len)[0]


def _tree(h5):
    out = []
    h5.visititems(lambda n, o: out.append(n) if hasattr(o, "shape") and len(o.shape) == 2 and "frequencyA" in n else None)
    return out


def _up(h5, group, name):
    """Procura `name` no grupo e nos grupos acima (coordenadas e projeção ficam no grupo da grade)."""
    g = group
    while g:
        if f"{g}/{name}" in h5:
            return f"{g}/{name}"
        g = g.rsplit("/", 1)[0] if "/" in g else ""
    raise KeyError(f"{name} não encontrado acima de {group}")


def _epsg(h5, group):
    ds = h5[_up(h5, group, "projection")]
    for k, v in ds.attrs.items():
        if "epsg" in k.lower():
            return int(np.atleast_1d(v)[0])
    return int(ds[()])


def identification(h5):
    """Metadados internos do arquivo (grupo identification): direção de visada, órbita, polígono, etc."""
    out = {}
    for root in ("science/LSAR/identification", "science/SSAR/identification"):
        if root in h5:
            for k, ds in h5[root].items():
                try:
                    if hasattr(ds, "shape") and (ds.shape == () or (len(ds.shape) == 1 and ds.shape[0] <= 4)):
                        v = ds[()]
                        v = v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)
                        if isinstance(v, list):
                            v = [x.decode() if isinstance(x, bytes) else x for x in v]
                        out[k] = v
                except Exception:
                    pass
            out["banda"] = "L" if "LSAR" in root else "S"
    return out


def read_layers(granule, grid, layers, linear=True):
    """Lê só a janela da área de cada camada e reprojeta para a grade comum.

    layers: {apelido: (nome_do_dataset, trecho_do_caminho_preferido_ou_None)}.
    linear=True descarta valores <= 0 (potência); False aceita negativos (fase, deslocamento)."""
    import earthaccess
    import h5py
    f = earthaccess.open([granule])[0]
    out = {}
    with h5py.File(f, "r") as h5:
        ident = identification(h5)
        paths = {k: _locate(h5, base, cont) for k, (base, cont) in layers.items()}
        first = next(iter(paths.values()))
        group = first.rsplit("/", 1)[0]
        x = h5[_up(h5, group, "xCoordinates")][()]
        y = h5[_up(h5, group, "yCoordinates")][()]
        epsg = _epsg(h5, group)
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        w, s, e, n = grid.bbox
        xs, ys = zip(*[tr.transform(a, b) for a, b in [(w, s), (w, n), (e, s), (e, n)]])
        dx, dy = x[1] - x[0], y[1] - y[0]
        pad = 3 * abs(dx)
        ix = np.where((x >= min(xs) - pad) & (x <= max(xs) + pad))[0]
        iy = np.where((y >= min(ys) - pad) & (y <= max(ys) + pad))[0]
        if ix.size == 0 or iy.size == 0:
            raise ValueError("A área não cai dentro desta cena")
        x0, x1, y0, y1 = ix.min(), ix.max() + 1, iy.min(), iy.max() + 1
        src_tr = Affine(dx, 0, x[x0] - dx / 2, 0, dy, y[y0] - dy / 2)
        # grade grossa (visão regional): média em blocos f×f lida em faixas, para não carregar
        # o quadro inteiro na memória (um quadro de 240 km a 10 m tem ~1 GB por camada)
        fct = max(1, int(grid.resolution // (2 * abs(dx))))
        for alias, pth in paths.items():
            ds = h5[pth]
            if ds.shape != (len(y), len(x)):          # camada noutra grade (ex.: coerência mais grossa)
                raise ValueError(f"{pth} tem forma {ds.shape}, diferente das coordenadas {(len(y), len(x))}")
            def clean(a):
                a = a.astype("float32")
                bad = ~np.isfinite(a) | (a <= 0 if linear else False)
                if not linear:
                    bad |= a == 0                      # 0 = sem dado nos produtos de fase
                a[bad] = np.nan
                return a
            if fct == 1:
                arr = clean(ds[y0:y1, x0:x1]); a_tr = src_tr
            else:
                ny, nx = (y1 - y0) // fct, (x1 - x0) // fct
                arr = np.full((ny, nx), np.nan, "float32")
                rows = max(fct, (2048 // fct) * fct)
                for r0 in range(0, ny * fct, rows):
                    blk = clean(ds[y0 + r0:y0 + min(r0 + rows, ny * fct), x0:x0 + nx * fct])
                    br = blk.shape[0] // fct
                    with np.errstate(invalid="ignore"):
                        arr[r0 // fct:r0 // fct + br] = np.nanmean(blk[:br * fct].reshape(br, fct, nx, fct), axis=(1, 3))
                a_tr = Affine(dx * fct, 0, x[x0] - dx / 2, 0, dy * fct, y[y0] - dy / 2)
            dst = np.full((grid.height, grid.width), np.nan, dtype="float32")
            reproject(arr, dst, src_transform=a_tr, src_crs=f"EPSG:{epsg}", src_nodata=np.nan,
                      dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                      resampling=Resampling.average)
            out[alias] = dst.astype("float64")
    meta = parse_name(granule["umm"]["GranuleUR"])
    meta.update({"arquivo": granule["umm"]["GranuleUR"], "epsg_produto": epsg,
                 "janela_linhas": [int(y0), int(y1)], "janela_colunas": [int(x0), int(x1)],
                 "caminhos_hdf5": paths, "espacamento_produto_m": float(abs(dx)), "fator_bloco": fct,
                 "identificacao": ident})
    return out, meta


def read_gcov_subset(granule, grid, layers=("HHHH", "HVHV")):
    return read_layers(granule, grid, {l: (l, None) for l in layers}, linear=True)


GUNW_LAYERS = {"fase": ("unwrappedPhase", "unwrappedInterferogram"),
               "coerencia": ("coherenceMagnitude", "unwrappedInterferogram")}
WAVELENGTH_L = 0.2385


def read_gunw_subset(granule, grid):
    """Deslocamento na linha de visada (cm) e coerência de um par GUNW, só no recorte.
    d = -fase · λ / (4π): fase em radianos; sinal positivo = aproximou do satélite (convenção declarada)."""
    arrs, meta = read_layers(granule, grid, GUNW_LAYERS, linear=False)
    arrs["deslocamento_cm"] = -arrs["fase"] * WAVELENGTH_L / (4 * np.pi) * 100.0
    return arrs, meta
