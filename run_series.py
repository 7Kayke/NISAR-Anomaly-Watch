# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""A valsa: série temporal completa do NISAR para uma área específica.

Uso:
    python run_series.py --case cases/norte_fire.json --out docs/results/norte_fire_serie
        [--start 2026-01-20] [--end 2026-11-14] [--max-dates 40] [--skip-truth] [--skip-dem]

Etapas: funil da série (com os metadados de TODOS os grânulos e onde cada um parou) -> leitura do
recorte de cada passagem (cache em cache/) -> classes de ritmo e quebra -> mapas, animação, curvas
com a chuva do NASA POWER -> validação contra a cicatriz do Sentinel-2 quando o caso tem evento.
"""
import argparse
import csv
import hashlib
import json
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from deltal import __version__, report
from deltal.change import boxcar_linear, to_db
from deltal.grid import make_grid
from deltal import series as S

DEEP, ELECTRIC, YONDER, YELLOW = report.DEEP, report.ELECTRIC, report.YONDER, report.YELLOW
BETA_END = "2026-01-20"
GAPS = [("2026-07-27", "2026-08-10", "falha permanente de dados"), ("2026-10-23", "2026-11-09", "pausa da produção provisória")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_cached(granule, grid, cache):
    from deltal import sources_nisar as sn
    name = granule["umm"]["GranuleUR"]
    h = hashlib.md5(json.dumps([grid.bbox, grid.resolution]).encode()).hexdigest()[:8]
    f = cache / f"{name}_{h}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return {"HHHH": z["HHHH"], "HVHV": z["HVHV"]}, json.loads(str(z["meta"]))
    arrs, meta = sn.read_gcov_subset(granule, grid)
    np.savez_compressed(f, HHHH=arrs["HHHH"].astype("float32"), HVHV=arrs["HVHV"].astype("float32"),
                        meta=json.dumps(meta, default=str))
    return arrs, meta


def s2_truth(grid, event_date):
    from deltal import collect
    from deltal.change import nbr, min_area_filter
    fun, best = collect.sentinel2(grid, event_date)
    _, pre, post, gap = best
    r0, c0, w0 = collect.sentinel2_load(pre, grid)
    r1, c1, w1 = collect.sentinel2_load(post, grid)
    with np.errstate(invalid="ignore"):
        dnbr = nbr(r0["B08"], r0["B12"]) - nbr(r1["B08"], r1["B12"])
        ok = c0 & c1 & ~w0 & ~w1 & np.isfinite(dnbr)
    t, _ = min_area_filter(ok & (dnbr > 0.1), int(10000 / grid.pixel_area_m2))
    return t, ok, {"antes": pre.day, "depois": post.day, "funil": fun.summary()}


def plot_classes(path, cls, title):
    cmap = ListedColormap([S.CLASS_COLORS[c] if c else "#FFFFFF" for c in range(7)])
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=110)
    ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax.set_axis_off(); ax.set_title(title, color=DEEP, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.CLASS_COLORS[c]) for c in range(1, 7)]
    ax.legend(handles, [S.CLASSES[c] for c in range(1, 7)], loc="lower left", fontsize=8, framealpha=0.85)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_when(path, when, dates, title):
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=110)
    m = np.ma.masked_where(when < 1, when)
    im = ax.imshow(m, cmap="plasma", vmin=1, vmax=len(dates) - 1, interpolation="nearest")
    ax.set_axis_off(); ax.set_title(title, color=DEEP, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ticks = np.unique(np.linspace(1, len(dates) - 1, min(8, len(dates) - 1)).astype(int))
    cb.set_ticks(ticks); cb.set_ticklabels([dates[i] for i in ticks])
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def plot_curves(path, dates, res, rain, event_date, title):
    d = [datetime.fromisoformat(x) for x in dates]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=110, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1.3]})
    a1.plot(d, res["curva_cena_hh"], "o-", color=ELECTRIC, label="HH (mediana da área)")
    a1.plot(d, res["curva_cena_hv"], "o-", color=YONDER, label="HV (mediana da área)")
    a1.set_ylabel("dB"); a1.set_title(title, color=DEEP, fontsize=11)
    if rain:
        ax2 = a1.twinx()
        rd = [datetime.strptime(k, "%Y%m%d") for k in rain]
        ax2.bar(rd, list(rain.values()), width=1.0, color="#7FB3D5", alpha=0.45, label="chuva (mm/dia)")
        ax2.set_ylabel("chuva mm/dia (NASA POWER)")
    hv_n, cls = res["hv_norm"], res["classe"]
    for c in (1, 2, 3, 4, 5):
        m = cls == c
        if m.sum() > 20:
            a2.plot(d, [np.nanmedian(s[m]) for s in hv_n], "o-", ms=3, color=S.CLASS_COLORS[c],
                    label=f"{S.CLASSES[c]} ({m.sum()} px)")
    a2.axhline(0, color="#999", lw=0.8); a2.set_ylabel("HV menos a cena (dB)")
    for ax in (a1, a2):
        for g0, g1, _ in GAPS:
            ax.axvspan(datetime.fromisoformat(g0), datetime.fromisoformat(g1), color="#ccc", alpha=0.35)
        if event_date:
            ax.axvline(datetime.fromisoformat(event_date), color="#FF6B00", ls="--", lw=1.2)
    a1.legend(loc="upper left", fontsize=8); a2.legend(loc="lower left", fontsize=7, ncol=2)
    fig.autofmt_xdate(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def animate(path, hv_db, dates, step=2):
    from PIL import Image, ImageDraw
    v = hv_db[np.isfinite(hv_db)]
    lo, hi = np.percentile(v, [2, 98])
    cm = plt.get_cmap("gray")
    frames = []
    for s, dt in zip(hv_db, dates):
        a = np.nan_to_num(np.clip((s[::step, ::step] - lo) / (hi - lo), 0, 1))
        im = Image.fromarray((cm(a)[..., :3] * 255).astype("uint8"))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 150, 18], fill=(7, 23, 63)); dr.text((5, 3), f"NISAR HV {dt}", fill=(234, 254, 7))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=600, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--max-dates", type=int, default=40)
    ap.add_argument("--skip-truth", action="store_true"); ap.add_argument("--skip-dem", action="store_true")
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path("cache"); cache.mkdir(exist_ok=True)
    cfg = case.get("serie", {})
    start = a.start or cfg.get("inicio", BETA_END)
    end = a.end or cfg.get("fim", date.today().isoformat())
    grid = make_grid(case["bbox"], case.get("resolucao_m", 20))
    px = grid.pixel_area_m2
    ev = (case.get("evento") or {}).get("data")
    ficha = {"nome": case["nome"], "gerado_em": datetime.now().isoformat(timespec="seconds"), "versao_pipeline": __version__,
             "area_bbox": case["bbox"], "periodo": [start, end], "evento": case.get("evento"),
             "grade": {"crs": grid.crs, "resolucao_m": grid.resolution, "largura": grid.width, "altura": grid.height},
             "metodo": S.__doc__.strip(), "parametros": S.PARAMS, "erros": {}}

    # 1) funil + metadados
    from deltal import sources_nisar as sn
    sn.login()
    log("funil da série NISAR ...")
    fun, chosen, meta, geoms = sn.funnel_series(grid, start, end)
    ficha["funil"] = fun.summary(); ficha["geometrias_que_cobrem"] = geoms
    keys = sorted({k for m in meta for k in m})
    with open(out / "metadados.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for m in meta:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in m.items()})
    (out / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"  {len(meta)} grânulos no catálogo, {len(chosen)} na série escolhida ({geoms})")
    if len(chosen) > a.max_dates:
        chosen = chosen[-a.max_dates:]
    if len(chosen) < 4:
        raise SystemExit("Menos de 4 passagens na mesma geometria: não dá para falar de ritmo aqui.")

    # 2) pilha
    hh, hv, dates, used = [], [], [], []
    for n, (snm, g) in enumerate(chosen, 1):
        name = g["umm"]["GranuleUR"]
        log(f"  lendo {n}/{len(chosen)} {name[:60]}...")
        try:
            arrs, m = read_cached(g, grid, cache)
        except Exception as e:
            ficha["erros"][name] = f"{type(e).__name__}: {e}"; traceback.print_exc(); continue
        hh.append(to_db(boxcar_linear(arrs["HHHH"], 3))); hv.append(to_db(boxcar_linear(arrs["HVHV"], 3)))
        dates.append(datetime.strptime(m["inicio"], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d"))
        used.append({"data": dates[-1], "arquivo": name, "colecao": snm, "crid": m.get("crid"), "ciclo": m.get("ciclo")})
    hh, hv = np.array(hh), np.array(hv)
    ficha["passagens_usadas"] = used

    # 3) máscaras
    valid = np.isfinite(np.nanmedian(hv, 0))
    land = valid & ~(np.nanmedian(hh, 0) < S.PARAMS["agua_db"])
    if not a.skip_dem:
        try:
            from deltal import collect
            from deltal.terrain import radar_masks
            _, z = collect.dem(grid)
            lay, sha, _ = radar_masks(z, grid.resolution, "nisar", sn.parse_name(chosen[0][1]["umm"]["GranuleUR"])["direcao"])
            ficha["relevo_mascarado_ha"] = round(float((lay | sha).sum() * px / 1e4), 2)
            land &= ~(lay | sha)
            valid &= ~(lay | sha)
        except Exception as e:
            ficha["erros"]["dem"] = f"{type(e).__name__}: {e}"

    # 4) análise
    log("classificando o ritmo ...")
    res = S.classify(hh, hv, land, valid=valid)
    truth = None
    if ev and not a.skip_truth:
        try:
            log("gabarito Sentinel-2 ...")
            truth, okt, info = s2_truth(grid, ev)
            ficha["gabarito_sentinel2"] = info | {"area_ha": round(float(truth.sum() * px / 1e4), 2)}
        except Exception as e:
            ficha["erros"]["gabarito"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
    ficha["resumo"] = S.summarize(res, dates, px, ev, truth)
    if truth is not None and "queda_evento" in res:
        from deltal.validate import metrics
        ficha["validacao_queda_no_evento"] = metrics(res["queda_evento"], truth, okt & land, px)

    # 5) chuva
    rain = None
    try:
        from deltal.sources_power import rain_between
        w, s_, e_, n_ = case["bbox"]
        rr = rain_between((w + e_) / 2, (s_ + n_) / 2, dates[0].replace("-", ""), dates[-1].replace("-", ""))
        rain = rr["diario_mm"]
        ficha["chuva_total_mm"] = rr["total_mm"]
        ficha["chuva_entre_passagens_mm"] = {
            f"{dates[i-1]}→{dates[i]}": round(sum(v for k, v in rain.items()
                                                  if dates[i-1].replace("-", "") < k <= dates[i].replace("-", "")), 1)
            for i in range(1, len(dates))}
    except Exception as e:
        ficha["erros"]["chuva"] = f"{type(e).__name__}: {e}"

    # 6) saídas
    log("figuras ...")
    plot_classes(out / "classes.png", res["classe"], f"Ritmo de cada pixel, {dates[0]} a {dates[-1]} ({len(dates)} passagens)")
    plot_when(out / "quando_mudou.png", res["quebra_indice"], dates, "Quando a mudança abrupta aconteceu")
    rgb, labels = S.temporal_rgb(hv, dates)
    report.overlay_png(out / "rgb_temporal.png", rgb, truth=truth,
                       title=f"HV em 3 tempos: R {labels[0]} | G {labels[1]} | B {labels[2]}")
    plot_curves(out / "curvas.png", dates, res, rain, ev, "O ritmo da área: radar e chuva")
    animate(out / "animacao_hv.gif", hv, dates)
    # série reduzida para clicar num ponto na interface (60 m, uint8)
    st = 3
    small = hv[:, ::st, ::st]
    lo, hi = -30.0, 0.0
    q = np.where(np.isfinite(small), np.clip((small - lo) / (hi - lo) * 254 + 1, 1, 255), 0).astype("uint8")
    q.tofile(out / "serie_hv_u8.bin")
    serie = {"datas": dates, "forma": list(q.shape), "escala_db": [lo, hi], "codificacao": "u8: 0=sem dado, 1..255 = lo..hi dB",
             "passo_pixels": st, "bbox": case["bbox"], "curva_cena_hv": res["curva_cena_hv"].tolist(), "curva_cena_hh": res["curva_cena_hh"].tolist(),
             "curvas_classes_hv_norm": {S.CLASSES[c]: [float(np.nanmedian(s[res["classe"] == c])) if (res["classe"] == c).any() else None
                                                       for s in res["hv_norm"]] for c in range(1, 6)},
             "chuva_mm": rain}
    (out / "serie.json").write_text(json.dumps(report._clean(serie), ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "arrays.npz", classe=res["classe"], quebra=res["quebra_indice"], delta=res["delta_db"])

    fic = report.write_ficha(out, ficha)
    lines = [f"# {case['nome']}: a valsa", "", f"{len(dates)} passagens do NISAR (banda L, GCOV) de {dates[0]} a {dates[-1]}.", "",
             "## Funil", ""] + [f"- {e['etapa']}: {e['n']}" + (f" ({e['aproveitamento_pct']}%)" if e['aproveitamento_pct'] is not None else "")
                                 + f" — {e['nota']}" for e in fic["funil"]["etapas"]]
    lines += ["", "## Ritmo da área", ""] + [f"- {k}: {v} ha" for k, v in fic["resumo"]["classes_ha"].items()]
    if fic["resumo"].get("quedas_por_intervalo_ha"):
        lines += ["", "## Quedas abruptas por intervalo", ""] + [f"- {k}: {v} ha" for k, v in fic["resumo"]["quedas_por_intervalo_ha"].items()]
    for k in ("queda_no_intervalo_do_evento_ha", "na_cicatriz_classe_queda_pct", "na_cicatriz_data_da_quebra_mais_comum",
              "na_cicatriz_quebra_no_intervalo_certo_pct"):
        if k in fic["resumo"]:
            lines.append(f"- {k.replace('_', ' ')}: {fic['resumo'][k]}")
    if fic.get("validacao_queda_no_evento"):
        m = fic["validacao_queda_no_evento"]
        lines += ["", f"Validação (queda no intervalo do evento contra a cicatriz Sentinel-2): precisão {m['precisao']}, "
                      f"sensibilidade {m['sensibilidade']}, F1 {m['f1']}, IoU {m['iou']}."]
    if fic.get("erros"):
        lines += ["", "## Erros", ""] + [f"- {k}: {v}" for k, v in fic["erros"].items()]
    (out / "resultado.md").write_text("\n".join(lines), encoding="utf-8")
    api = Path("docs/api/v1/series"); api.mkdir(parents=True, exist_ok=True)
    (api / f"{Path(a.case).stem}.json").write_text(json.dumps({"id": Path(a.case).stem, "ficha": fic,
        "arquivos": sorted(f"{out.as_posix()}/{p.name}" for p in out.iterdir() if p.suffix != ".npz")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log("pronto")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
