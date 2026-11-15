# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Reconhecimento: quais pontos de uma região têm os melhores dados NISAR (só metadados, nada é baixado).

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
    ap.add_argument("--passo", type=float, default=0.25, help="espaçamento da grade em graus")
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
    grid = []
    for lat in np.arange(bbox[1] + a.passo / 2, bbox[3], a.passo):
        for lon in np.arange(bbox[0] + a.passo / 2, bbox[2], a.passo):
            if reg.get("mascara_poligono") and not shape(reg["mascara_poligono"]).contains(Point(lon, lat)):
                continue
            r = {"nome": f"grade {lat:.2f},{lon:.2f}", "lat": round(float(lat), 3), "lon": round(float(lon), 3), "por_que": "grade"}
            r.update(stats_at(lon, lat, items)); grid.append(r)
    score(cands + grid)
    cands.sort(key=lambda r: -r["nota"]); grid.sort(key=lambda r: -r["nota"])
    fields = ["nota", "nome", "lat", "lon", "gcov", "gcov_dupla", "gcov_dupla_pos_beta", "geometrias_boas", "maior_buraco_dias",
              "gunw", "sme2", "primeira", "ultima", "geometrias", "por_que"]
    for nm, rows in (("candidatos.csv", cands), ("grade.csv", grid)):
        with open(out / nm, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({**r, "geometrias": json.dumps(r["geometrias"])})
    # mapa
    fig, ax = plt.subplots(figsize=(9, 10), dpi=110)
    for m in items:
        if m["produto"] == "GCOV":
            x, y = m["shape"].exterior.xy; ax.plot(x, y, color="#2E96F5", lw=0.3, alpha=0.25)
    if grid:
        sc = ax.scatter([r["lon"] for r in grid], [r["lat"] for r in grid], c=[r["nota"] for r in grid], cmap="viridis",
                        s=60, marker="s", vmin=0, vmax=100)
        fig.colorbar(sc, ax=ax, fraction=0.04, label="nota de cobertura (0-100)")
    for r in cands:
        ax.plot(r["lon"], r["lat"], "*", ms=14, color="#FF6B00", mec="k")
        ax.annotate(f"{r['nome']} ({r['nota']})", (r["lon"], r["lat"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3]); ax.set_aspect("equal")
    ax.set_title(f"{reg['nome']}: onde o NISAR tem mais dados", color=report.DEEP)
    fig.savefig(out / "mapa_cobertura.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    lines = [f"# {reg['nome']}: reconhecimento de cobertura NISAR", "", f"{start} a {end}. Coleções: {', '.join(cols)}.", "",
             "| nota | ponto | GCOV duplas (pós-beta) | geometrias com ≥4 | maior buraco (dias) | GUNW | SME2 |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['nota']} | {r['nome']} ({r['lat']}, {r['lon']}) | {r['gcov_dupla_pos_beta']} | {r['geometrias_boas']} | "
              f"{r['maior_buraco_dias']} | {r['gunw']} | {r['sme2']} |" for r in cands]
    lines += ["", "## Melhores células da grade", ""] + [f"- {r['nota']}: ({r['lat']}, {r['lon']}) duplas {r['gcov_dupla_pos_beta']}, "
                                                        f"geometrias {r['geometrias']}, buraco {r['maior_buraco_dias']} d" for r in grid[:10]]
    (out / "resultado.md").write_text("\n".join(lines), encoding="utf-8")
    log("pronto"); print("\n".join(lines))


if __name__ == "__main__":
    main()
