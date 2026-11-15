# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Malha do NISAR: as células onde quadros ascendentes e descendentes se cruzam (só metadados).

O mapa não usa grade inventada: usa os quadros fixos do próprio NISAR (pista, direção, quadro).
Cada quadro ascendente (madrugada, olhando para um lado) cruza com quadros descendentes (noite, olhando
para o outro). Cada cruzamento A x D é uma CÉLULA vista de dois ângulos: é ali que se triangula.

Uso:
    python run_scout.py --regiao cases/scout_pantanal.json --out docs/results/scout_pantanal

Faz UMA busca por coleção útil (GCOV, GUNW, SME2) na região inteira e depois, para cada ponto candidato
e para uma grade regular, conta o que cobre aquele ponto:
  - passagens GCOV únicas (uma versão por passagem), quantas com polarização dupla (HH+HV)
  - geometrias com >= 4 passagens duplas (séries comparáveis)
  - maior buraco sem passagem (dias), desde 20/01/2026
  - pares GUNW e arquivos SME2
Nota (0 a 100) = 35% passagens duplas + 20% geometrias boas + 20% menor buraco + 15% GUNW + 10% SME2,
cada item relativo ao melhor ponto da região.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape
from shapely.prepared import prep

from deltal import report

USEFUL = ("GCOV", "GUNW", "SME2")
LAUNCH, BETA_END = "2025-07-30", "2026-01-20"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(bbox, start, end):
    import earthaccess
    from deltal.inventory import discover_collections, _maturity, _product
    from deltal.sources_nisar import _umm_geometry, parse_name
    cols = [c["short_name"] for c in discover_collections() if any(k in c["short_name"].upper() for k in USEFUL)]
    best = {}
    for sn in cols:
        t0 = time.time()
        grans = earthaccess.search_data(short_name=sn, bounding_box=tuple(bbox), temporal=(start, end))
        log(f"  {sn}: {len(grans)} arquivos ({time.time() - t0:.0f} s)")
        for g in grans:
            name = g["umm"]["GranuleUR"]
            m = parse_name(name)
            geo = _umm_geometry(g)
            if not geo or m.get("pista") is None:
                continue
            m["produto"] = _product(sn, m)
            k = (m["produto"], m.get("ciclo"), m["pista"], m["direcao"], m["quadro"], (m.get("inicio") or "")[:8],
                 (m.get("inicio_par2") or "")[:8])
            rank = (_maturity(sn), "~" + (m.get("crid") or ""))
            if k not in best or rank < best[k][0]:
                m["poly"] = prep(shape(geo)); m["shape"] = shape(geo)
                best[k] = (rank, m)
    return cols, [v[1] for v in best.values()]


def stats_at(lon, lat, items):
    pt = Point(lon, lat)
    hit = [m for m in items if m["poly"].contains(pt)]
    gc = [m for m in hit if m["produto"] == "GCOV"]
    dual = [m for m in gc if (m.get("polarizacao") or "").startswith("D") or (m.get("polarizacao") or "") in ("QPQP", "QQ")]
    geos = defaultdict(int)
    for m in dual:
        if (m.get("inicio") or "") >= BETA_END.replace("-", ""):
            geos[f"{m['pista']:03d}_{m['direcao']}_{m['quadro']:03d}"] += 1
    days = sorted({m["inicio"][:8] for m in gc if m.get("inicio") and m["inicio"][:8] >= BETA_END.replace("-", "")})
    d = [datetime.strptime(x, "%Y%m%d") for x in days]
    gap = max([(b - a).days for a, b in zip(d, d[1:])] + [(datetime.now() - d[-1]).days] if d else [999])
    return {"gcov": len(gc), "gcov_dupla": len(dual), "gcov_dupla_pos_beta": sum(geos.values()),
            "geometrias_boas": sum(1 for v in geos.values() if v >= 4), "geometrias": dict(geos),
            "maior_buraco_dias": gap, "gunw": sum(1 for m in hit if m["produto"] == "GUNW"),
            "sme2": sum(1 for m in hit if m["produto"] == "SME2"),
            "primeira": min((m["inicio"][:8] for m in gc if m.get("inicio")), default=None),
            "ultima": max((m["inicio"][:8] for m in gc if m.get("inicio")), default=None)}


def score(rows):
    def rel(k, inv=False):
        v = np.array([r[k] for r in rows], float)
        if inv:
            v = v.max() - v
        return v / v.max() if v.max() > 0 else v * 0
    s = 35 * rel("gcov_dupla_pos_beta") + 20 * rel("geometrias_boas") + 20 * rel("maior_buraco_dias", True) \
        + 15 * rel("gunw") + 10 * rel("sme2")
    for r, v in zip(rows, s):
        r["nota"] = round(float(v), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regiao", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-area-km2", type=float, default=100.0, help="célula mínima (km²)")
    a = ap.parse_args()
    reg = json.loads(Path(a.regiao).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from deltal import sources_nisar as sn
    sn.login()
    bbox = reg["bbox"]; start = reg.get("inicio", LAUNCH); end = reg.get("fim", date.today().isoformat())
    log(f"buscando GCOV, GUNW e SME2 em {bbox}, {start} a {end} ...")
    cols, items = fetch(bbox, start, end)
    log(f"{len(items)} passagens/pares únicos")
    cands = []
    for c in reg["candidatos"]:
        r = {"nome": c["nome"], "lat": c["lat"], "lon": c["lon"], "por_que": c.get("por_que", "")}
        r.update(stats_at(c["lon"], c["lat"], items)); cands.append(r)
    # quadros do NISAR: um polígono por (pista, direção, quadro) = o maior entre as passagens (quadro cheio)
    from shapely.geometry import box
    region = box(*bbox)
    frames = {}
    for m in items:
        if m["produto"] != "GCOV":
            continue
        k = (m["pista"], m["direcao"], m["quadro"])
        if k not in frames or m["shape"].area > frames[k].area:
            frames[k] = m["shape"]
    asc = {k: v for k, v in frames.items() if k[1] == "A"}
    des = {k: v for k, v in frames.items() if k[1] == "D"}
    log(f"malha: {len(asc)} quadros ascendentes x {len(des)} descendentes")
    grid, cells = [], []
    km2 = lambda g, lat: g.area * (111.32 ** 2) * np.cos(np.radians(lat))
    for ka, pa in asc.items():
        for kd, pd in des.items():
            c = pa.intersection(pd).intersection(region)
            if c.is_empty or c.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            rp = c.representative_point()
            if km2(c, rp.y) < a.min_area_km2:
                continue
            cid = f"A{ka[0]:03d}-{ka[2]:03d} x D{kd[0]:03d}-{kd[2]:03d}"
            r = {"nome": cid, "lat": round(rp.y, 4), "lon": round(rp.x, 4), "area_km2": round(km2(c, rp.y)),
                 "por_que": "cruzamento de quadros NISAR"}
            r.update(stats_at(rp.x, rp.y, items)); grid.append(r); cells.append((r, c))
    for r in cands:
        pt = Point(r["lon"], r["lat"])
        r["celula"] = next((x["nome"] for x, c in cells if c.contains(pt)), "fora dos cruzamentos")
    score(cands + grid)
    cands.sort(key=lambda r: -r["nota"]); grid.sort(key=lambda r: -r["nota"])
    fields = ["nota", "nome", "lat", "lon", "area_km2", "celula", "gcov", "gcov_dupla", "gcov_dupla_pos_beta", "geometrias_boas", "maior_buraco_dias",
              "gunw", "sme2", "primeira", "ultima", "geometrias", "por_que"]
    # malha para o aplicativo: quadros e células como GeoJSON
    from shapely.geometry import mapping
    feats = [{"type": "Feature", "geometry": mapping(f.simplify(0.001)),
              "properties": {"tipo": "quadro", "id": f"{k[1]}{k[0]:03d}-{k[2]:03d}", "pista": k[0], "direcao": k[1], "quadro": k[2]}}
             for k, f in frames.items()]
    for r, c in cells:
        props = {k: r[k] for k in ("nome", "nota", "lat", "lon", "area_km2", "gcov", "gcov_dupla", "gcov_dupla_pos_beta",
                                   "geometrias_boas", "maior_buraco_dias", "gunw", "sme2", "primeira", "ultima", "geometrias")}
        feats.append({"type": "Feature", "geometry": mapping(c.simplify(0.001)), "properties": {"tipo": "celula", **props}})
    for r in cands:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": {"tipo": "lugar", "nome": r["nome"], "por_que": r["por_que"], "nota": r["nota"],
                                     "celula": r.get("celula"), "geometrias": r["geometrias"]}})
    (out / "malha.geojson").write_text(json.dumps({"type": "FeatureCollection", "regiao": reg["nome"], "bbox": bbox,
                                                   "periodo": [start, end], "colecoes": cols, "features": feats},
                                                  ensure_ascii=False, default=str), encoding="utf-8")
    for nm, rows in (("candidatos.csv", cands), ("celulas.csv", grid)):
        with open(out / nm, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({**r, "geometrias": json.dumps(r["geometrias"])})
    # mapa
    fig, ax = plt.subplots(figsize=(9, 10), dpi=110)
    from matplotlib import cm, colors as mcolors
    norm = mcolors.Normalize(0, 100); cmap = plt.get_cmap("viridis")
    for r, c in cells:
        for poly in (c.geoms if c.geom_type == "MultiPolygon" else [c]):
            x, y = poly.exterior.xy; ax.fill(x, y, color=cmap(norm(r["nota"])), alpha=0.75, lw=0)
    for k, f in frames.items():
        x, y = f.exterior.xy
        ax.plot(x, y, color="#0042A6" if k[1] == "A" else "#B5179E", lw=0.8, alpha=0.6)
        ax.annotate(f"{k[1]}{k[0]:03d}-{k[2]:03d}", (f.centroid.x, f.centroid.y), fontsize=6,
                    color="#0042A6" if k[1] == "A" else "#B5179E", ha="center")
    for r, c in sorted(cells, key=lambda t: -t[0]["nota"])[:5]:
        ax.annotate(f"{r['nota']}", (r["lon"], r["lat"]), fontsize=8, weight="bold", ha="center", color="white")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, label="nota da célula (0-100)")
    ax.plot([], [], color="#0042A6", label="quadros ascendentes (madrugada)")
    ax.plot([], [], color="#B5179E", label="quadros descendentes (noite)"); ax.legend(loc="lower left", fontsize=7)
    for r in cands:
        ax.plot(r["lon"], r["lat"], "*", ms=14, color="#FF6B00", mec="k")
        ax.annotate(f"{r['nome']} ({r['nota']})", (r["lon"], r["lat"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3]); ax.set_aspect("equal")
    ax.set_title(f"{reg['nome']}: malha do NISAR e cruzamentos A x D", color=report.DEEP)
    fig.savefig(out / "mapa_cobertura.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    lines = [f"# {reg['nome']}: malha do NISAR", "", f"{start} a {end}. {len(asc)} quadros ascendentes, {len(des)} descendentes, "
             f"{len(cells)} cruzamentos. Coleções: {', '.join(cols)}.", "",
             "| nota | célula (A x D) | centro | km² | GCOV duplas | geometrias ≥4 | maior buraco (dias) | GUNW | SME2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['nota']} | {r['nome']} | {r['lat']}, {r['lon']} | {r['area_km2']} | {r['gcov_dupla_pos_beta']} | {r['geometrias_boas']} | "
              f"{r['maior_buraco_dias']} | {r['gunw']} | {r['sme2']} |" for r in grid[:15]]
    lines += ["", "## Lugares conhecidos e a célula onde caem", ""] + [f"- {r['nome']}: {r['celula']} (nota do ponto {r['nota']})" for r in cands]
    (out / "resultado.md").write_text("\n".join(lines), encoding="utf-8")
    log("pronto"); print("\n".join(lines))


if __name__ == "__main__":
    main()
