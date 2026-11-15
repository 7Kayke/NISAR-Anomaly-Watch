# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Inventário de TODOS os produtos NISAR sobre uma área: o funil completo de metadados.

As coleções são descobertas no próprio catálogo da NASA (CMR), não ficam fixas no código: tudo o que
tiver nome começando por NISAR_ entra (L0B, RSLC, GSLC, GCOV, RIFG, RUNW, GUNW, ROFF, GOFF, SME2...).

Filtros (os mesmos do Vertex, na ordem do funil), por produto:
  catalogo          área + período
  cobre_area        o polígono do arquivo cobre >= min_cover da área
  mesmo_quadro      mesma pista, direção e quadro da série (quando o produto tem quadro)
  uma_por_passagem  uma versão por passagem/par: validada > provisória > beta; CRID mais novo
Cada arquivo guarda a etapa onde parou e o motivo.
"""
from collections import defaultdict
from datetime import datetime

from .funnel import Funnel, coverage
from .sources_nisar import _umm_geometry, _umm_meta, parse_name

MATURITY_ORDER = ("V1", "PROVISIONAL", "BETA")


def discover_collections():
    import earthaccess
    found = {}
    for kw in ("NISAR",):
        for c in earthaccess.search_datasets(keyword=kw):
            try:
                sn = c["umm"]["ShortName"]
            except (KeyError, TypeError):
                continue
            if sn.upper().startswith("NISAR"):
                found[sn] = {"short_name": sn, "titulo": c["umm"].get("EntryTitle"),
                             "versao": c["umm"].get("Version"), "concept_id": c.get("meta", {}).get("concept-id")}
    return sorted(found.values(), key=lambda d: d["short_name"])


def _maturity(short_name):
    up = short_name.upper()
    for i, k in enumerate(("PROVISIONAL", "BETA")):
        if k in up:
            return i + 1
    return 0


def _product(short_name, meta):
    if meta.get("produto"):
        return meta["produto"]
    parts = short_name.split("_")
    return parts[2] if len(parts) > 2 else short_name


def inventory(bbox, start, end, geometry=None, min_cover=0.9, collections=None, log=print):
    import earthaccess
    cols = collections or discover_collections()
    log(f"    {len(cols)} coleções NISAR no catálogo: {', '.join(c['short_name'] for c in cols)}")
    rows, by_prod = [], defaultdict(list)
    for c in cols:
        sn = c["short_name"]
        try:
            grans = earthaccess.search_data(short_name=sn, bounding_box=tuple(bbox), temporal=(start, end))
        except Exception as e:
            log(f"    {sn}: erro na busca ({type(e).__name__}: {e})")
            continue
        log(f"    {sn}: {len(grans)} arquivos")
        for g in grans:
            m = _umm_meta(sn, g)
            m["produto"] = _product(sn, m)
            m["maturidade"] = ("validado", "provisório", "beta")[_maturity(sn)]
            geo = _umm_geometry(g)
            m["cobertura_area"] = round(coverage(geo, bbox), 3) if geo else None
            m["etapa_final"], m["motivo"] = "catalogo", ""
            rows.append(m)
            by_prod[m["produto"]].append(m)

    funnels = []
    for prod, ms in sorted(by_prod.items()):
        fun = Funnel(f"NISAR {prod}", ", ".join(sorted({m["colecao_cmr"] for m in ms})))
        fun.add("catalogo", ms, f"{start} a {end}")
        cov = []
        for m in ms:
            if m["cobertura_area"] is None or m["cobertura_area"] >= min_cover:
                m["etapa_final"] = "cobre_area"; cov.append(m)
            else:
                m["motivo"] = f"cobre só {m['cobertura_area']} da área"
        fun.add("cobre_area", cov, f"cobre >= {int(min_cover * 100)}% da área (sem polígono: mantido)")
        same = []
        for m in cov:
            if geometry and m.get("pista") is not None and (m["pista"], m["direcao"], m["quadro"]) != tuple(geometry):
                m["motivo"] = f"quadro {m['pista']:03d}_{m['direcao']}_{m['quadro']:03d} diferente de {geometry}"
            else:
                m["etapa_final"] = "mesmo_quadro"; same.append(m)
        fun.add("mesmo_quadro", same, f"pista/direção/quadro = {geometry}" if geometry else "sem quadro fixo")
        best = {}
        for m in same:
            key = (m.get("ciclo"), m.get("pista"), m.get("direcao"), m.get("quadro"), m.get("inicio_par2"),
                   m.get("inicio") if m.get("ciclo") is None else None)
            rank = (_maturity(m["colecao_cmr"]), "" if not m.get("crid") else "~" + m["crid"])
            cur = best.get(key)
            if cur is None or rank < cur[0]:
                best[key] = (rank, m)
        keep = [v[1] for v in best.values()]
        ids = {id(m) for m in keep}
        for m in same:
            if id(m) in ids:
                m["etapa_final"] = "uma_por_passagem"
            else:
                m["motivo"] = "mesma passagem já existe em versão preferida"
        fun.add("uma_por_passagem", keep, "validado > provisório > beta; CRID mais novo")
        funnels.append(fun.summary())
    rows.sort(key=lambda m: (m.get("inicio") or "", m["produto"], m["arquivo"]))
    return rows, funnels, cols


def calendar_png(path, rows, start, end, gaps=(), title="Produtos NISAR disponíveis por passagem"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keep = [m for m in rows if m["etapa_final"] == "uma_por_passagem" and m.get("inicio")]
    prods = sorted({m["produto"] for m in keep}) or ["(nenhum)"]
    colors = {"validado": "#0042A6", "provisório": "#2E96F5", "beta": "#B0B7C3"}
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(prods) + 1.6), dpi=110)
    for g0, g1, _ in gaps:
        ax.axvspan(datetime.fromisoformat(g0), datetime.fromisoformat(g1), color="#ddd", alpha=0.6)
    for i, p in enumerate(prods):
        for m in [x for x in keep if x["produto"] == p]:
            d = datetime.strptime(m["inicio"], "%Y%m%dT%H%M%S")
            ax.plot(d, i, "o", ms=7, color=colors[m["maturidade"]])
            if m.get("inicio_par2"):
                d2 = datetime.strptime(m["inicio_par2"], "%Y%m%dT%H%M%S")
                ax.plot([d, d2], [i, i], "-", lw=1, color=colors[m["maturidade"]], alpha=0.5)
    ax.set_yticks(range(len(prods))); ax.set_yticklabels(prods)
    ax.set_xlim(datetime.fromisoformat(start), datetime.fromisoformat(end))
    for k, c in colors.items():
        ax.plot([], [], "o", color=c, label=k)
    ax.legend(loc="upper left", fontsize=8, ncol=3); ax.set_title(title, color="#07173F", fontsize=11)
    ax.grid(axis="x", alpha=0.3); fig.autofmt_xdate()
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)
