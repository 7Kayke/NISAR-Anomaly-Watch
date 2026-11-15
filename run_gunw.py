# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Interferometria (GUNW) do quadrante: deslocamento do chão e coerência, par a par.

Uso sozinho:
    python run_gunw.py --case cases/cerrado_matopiba.json --out docs/results/cerrado_matopiba/gunw [--max-pairs 30]

- deslocamento na linha de visada (cm) = -fase desdobrada · λ/(4π), RELATIVO à mediana da área
  (a fase desdobrada tem uma constante arbitrária; sem ponto de referência em solo estável)
- coerência (0 a 1): o quanto o "padrão fino" do chão continuou igual entre as duas datas.
  Alta = estável (solo exposto, rocha, cidade). Baixa = algo mexeu (lavoura, vegetação, água, obra).
"""
import argparse
import hashlib
import json
import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deltal import report
from deltal.funnel import coverage


def log(m):
    from run_series import log as _l
    _l(m)


def _key(m):
    return (m.get("pista"), m.get("direcao"), m.get("quadro"), m.get("inicio", "")[:8], (m.get("inicio_par2") or "")[:8])


def search(grid, start, end, min_cover=0.5):
    import earthaccess
    from deltal.inventory import discover_collections, _maturity
    from deltal.sources_nisar import _umm_geometry, parse_name
    cols = [c["short_name"] for c in discover_collections() if "GUNW" in c["short_name"].upper()]
    best = {}
    n_cat = 0
    for sn in cols:
        for g in earthaccess.search_data(short_name=sn, bounding_box=tuple(grid.bbox), temporal=(start, end)):
            n_cat += 1
            geo = _umm_geometry(g)
            if geo and coverage(geo, grid.bbox) < min_cover:
                continue
            m = parse_name(g["umm"]["GranuleUR"])
            if not m.get("inicio_par2"):
                continue
            k = _key(m)
            rank = (_maturity(sn), "~" + (m.get("crid") or ""))
            if k not in best or rank < best[k][0]:
                best[k] = (rank, sn, g, m)
    return cols, n_cat, [v[1:] for v in best.values()]


def run(case, out, grid, start, end, max_pairs=30):
    from deltal import export
    from deltal import sources_nisar as sn
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cache = Path("cache"); cache.mkdir(exist_ok=True)
    log("GUNW: buscando pares ...")
    cols, n_cat, pairs = search(grid, start, end)
    log(f"  coleções {cols}: {n_cat} no catálogo, {len(pairs)} pares únicos cobrindo a área")
    if not pairs:
        return ["Nenhum par GUNW cobre a área no período."]
    def span(p):
        a = datetime.strptime(p[2]["inicio"][:8], "%Y%m%d"); b = datetime.strptime(p[2]["inicio_par2"][:8], "%Y%m%d")
        return a, b
    pairs.sort(key=lambda p: (abs((span(p)[1] - span(p)[0]).days), span(p)[0]))
    pairs = sorted(pairs[:max_pairs], key=lambda p: span(p)[0])
    rows, cohs, disps = [], [], []
    h = hashlib.md5(json.dumps([grid.bbox, grid.resolution]).encode()).hexdigest()[:8]
    for i, (snm, g, m) in enumerate(pairs, 1):
        name = g["umm"]["GranuleUR"]
        a, b = span((snm, g, m))
        log(f"  par {i}/{len(pairs)} {a:%Y-%m-%d} -> {b:%Y-%m-%d} ({m['pista']:03d}_{m['direcao']}_{m['quadro']:03d})")
        f = cache / f"{name}_{h}.npz"
        try:
            if f.exists():
                z = np.load(f); arrs = {k: z[k] for k in ("deslocamento_cm", "coerencia")}
            else:
                arrs, meta = sn.read_gunw_subset(g, grid)
                np.savez_compressed(f, deslocamento_cm=arrs["deslocamento_cm"].astype("float32"),
                                    coerencia=arrs["coerencia"].astype("float32"))
        except Exception as e:
            log(f"    falhou: {type(e).__name__}: {e}"); traceback.print_exc()
            rows.append({"arquivo": name, "erro": f"{type(e).__name__}: {e}"}); continue
        d = arrs["deslocamento_cm"] - np.nanmedian(arrs["deslocamento_cm"])
        c = arrs["coerencia"]
        tag = f"{a:%Y%m%d}_{b:%Y%m%d}_{m['pista']:03d}{m['direcao']}"
        export._cog(out / f"par_{tag}.tif", [d, c], grid, ["deslocamento_cm_relativo", "coerencia"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), dpi=100)
        im0 = ax[0].imshow(d, cmap="RdBu_r", vmin=-5, vmax=5); ax[0].set_title(f"Deslocamento LOS (cm, relativo) {a:%d/%m}→{b:%d/%m}", fontsize=10)
        im1 = ax[1].imshow(c, cmap="viridis", vmin=0, vmax=1); ax[1].set_title("Coerência (0 = mudou, 1 = igual)", fontsize=10)
        for x, im in zip(ax, (im0, im1)):
            x.set_axis_off(); fig.colorbar(im, ax=x, fraction=0.046)
        fig.savefig(out / f"par_{tag}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
        rows.append({"arquivo": name, "colecao": snm, "data1": f"{a:%Y-%m-%d}", "data2": f"{b:%Y-%m-%d}",
                     "dias": (b - a).days, "geometria": f"{m['pista']:03d}_{m['direcao']}_{m['quadro']:03d}",
                     "coerencia_mediana": round(float(np.nanmedian(c)), 3),
                     "area_coerencia_baixa_pct": round(100 * float(np.nanmean(c < 0.3)), 1),
                     "deslocamento_p95_abs_cm": round(float(np.nanpercentile(np.abs(d), 95)), 2)})
        cohs.append(c); disps.append(d)
    ok = [r for r in rows if "erro" not in r]
    if cohs:
        mc = np.nanmean(np.array(cohs), 0)
        report.save_png(out / "coerencia_media.png", mc, cmap="viridis", vmin=0, vmax=1,
                        title="Valsa invisível: o que se mantém igual (coerência média dos pares)")
        export._cog(out / "coerencia_media.tif", [mc], grid, ["coerencia_media"])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
        for r in ok:
            d1, d2 = datetime.fromisoformat(r["data1"]), datetime.fromisoformat(r["data2"])
            ax.plot([d1, d2], [r["coerencia_mediana"]] * 2, "-", lw=2, color=report.ELECTRIC)
        ax.set_ylim(0, 1); ax.set_ylabel("coerência mediana da área"); ax.set_title("Coerência de cada par ao longo do ano", color=report.DEEP)
        fig.autofmt_xdate(); fig.savefig(out / "coerencia_no_tempo.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    (out / "pares.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"{len(ok)} pares lidos de {len(pairs)} ({n_cat} arquivos GUNW no catálogo).", "",
             "| Par | dias | geometria | coerência mediana | área com coerência < 0,3 | |desloc.| p95 (cm) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['data1']} → {r['data2']} | {r['dias']} | {r['geometria']} | {r['coerencia_mediana']} | {r['area_coerencia_baixa_pct']}% | {r['deslocamento_p95_abs_cm']} |" for r in ok]
    lines += [f"- falhou: {r['arquivo']}: {r['erro']}" for r in rows if "erro" in r]
    (out / "resultado.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def main():
    from deltal.grid import make_grid
    from run_series import BETA_END, fit_to_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=30); ap.add_argument("--res", type=float)
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    from deltal import sources_nisar as sn
    sn.login()
    start = case.get("serie", {}).get("inicio", BETA_END); end = case.get("serie", {}).get("fim", date.today().isoformat())
    bbox = case["bbox"]
    if case.get("nisar_quadro"):
        q = case["nisar_quadro"]; bbox, _ = fit_to_frame(bbox, start, end, (int(q["pista"]), q["direcao"], int(q["quadro"])), sn)
    grid = make_grid(bbox, a.res or case.get("resolucao_m", 20))
    print("\n".join(run(case, a.out, grid, start, end, a.max_pairs)))


if __name__ == "__main__":
    main()
