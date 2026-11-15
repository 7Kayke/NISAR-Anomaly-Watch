# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""A história completa de UM PONTO no NISAR: do lançamento (30/07/2025) até hoje.

Uso:
    python run_point.py --lat -7.40 --lon -47.50 --nome "Rio Tocantins" --out docs/results/ponto_tocantins
        [--inicio 2025-07-30] [--fim AAAA-MM-DD] [--raio-m 500] [--sem-leitura]

1. inventário: TODAS as coleções NISAR, todos os arquivos cujo polígono contém o ponto, com os metadados
   do catálogo e a etapa em que cada um parou (catálogo -> contém o ponto -> uma versão por passagem)
2. GCOV: para cada passagem de cada geometria, HH e HV no círculo de raio --raio-m em volta do ponto
   (e uma janela de 3 km para as miniaturas). Geometrias diferentes NÃO se misturam: cada pista/direção
   olha com outro ângulo, então cada uma tem a sua curva.
3. GUNW: coerência no ponto e deslocamento do ponto RELATIVO ao anel de 1 a 3 km em volta
4. clima diário do NASA POWER no ponto (chuva, temperatura, umidade do solo)
5. quebras e tendência na curva de cada geometria (mesmo método dos mapas, aplicado a 1 pixel)
"""
import argparse
import csv
import json
import time
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape

from deltal import report
from deltal import series as S
from deltal.change import to_db
from deltal.grid import make_grid

LAUNCH = "2025-07-30"
BETA_END = "2026-01-20"
GAPS = [("2026-07-27", "2026-08-10", "falha permanente de dados"), ("2026-10-23", "2026-11-09", "pausa da produção provisória")]
LAMBDA_L = 0.2385
USEFUL = ("GCOV", "GUNW", "SME2")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _geo_key(m):
    return f"{m['pista']:03d}_{m['direcao']}_{m['quadro']:03d}" if m.get("pista") is not None else "sem_quadro"


def inventory_point(lon, lat, start, end):
    import earthaccess
    from deltal.inventory import discover_collections, _maturity, _product
    from deltal.sources_nisar import _umm_geometry, _umm_meta
    pt = Point(lon, lat)
    d = 0.001
    # só os produtos que mudam o resultado no ponto:
    #   GCOV = brilho do radar (vegetação, água, umidade); GUNW = coerência e movimento do chão;
    #   SME2 = umidade do solo do próprio NISAR. Os demais (L0B, RSLC, GSLC, RIFG, RUNW, ROFF, GOFF)
    #   são versões brutas/intermediárias da mesma informação ou servem a movimentos grandes (geleiras).
    cols = [c for c in discover_collections() if any(k in c["short_name"].upper() for k in USEFUL)]
    log(f"{len(cols)} coleções NISAR úteis: {', '.join(c['short_name'] for c in cols)}")
    rows, objs = [], {}
    for c in cols:
        sn = c["short_name"]
        try:
            grans = earthaccess.search_data(short_name=sn, bounding_box=(lon - d, lat - d, lon + d, lat + d),
                                            temporal=(start, end))
        except Exception as e:
            log(f"  {sn}: erro {type(e).__name__}: {e}"); continue
        log(f"  {sn}: {len(grans)} arquivos")
        for g in grans:
            m = _umm_meta(sn, g)
            m["produto"] = _product(sn, m)
            m["maturidade"] = ("validado", "provisório", "beta")[_maturity(sn)]
            if m.get("inicio") and m["inicio"][:8] < BETA_END.replace("-", ""):
                m["periodo"] = "beta (antes de 20/01/2026)"
            geo = _umm_geometry(g)
            m["contem_ponto"] = bool(geo and shape(geo).buffer(1e-9).contains(pt)) if geo else None
            m["etapa_final"], m["motivo"] = "catalogo", ""
            rows.append(m); objs[m["arquivo"]] = (sn, g)
    # contém o ponto
    inside = []
    for m in rows:
        if m["contem_ponto"] is False:
            m["motivo"] = "o polígono do arquivo não contém o ponto (só encosta na caixa de busca)"
        else:
            m["etapa_final"] = "contem_ponto"; inside.append(m)
    # uma versão por passagem / par
    best = {}
    for m in inside:
        k = (m["produto"], m.get("ciclo"), _geo_key(m), (m.get("inicio") or "")[:8], (m.get("inicio_par2") or "")[:8])
        rank = (_maturity(m["colecao_cmr"]), "~" + (m.get("crid") or ""))
        if k not in best or rank < best[k][0]:
            best[k] = (rank, m)
    keep = {id(v[1]) for v in best.values()}
    for m in inside:
        if id(m) in keep:
            m["etapa_final"] = "uma_por_passagem"
        else:
            m["motivo"] = "mesma passagem já existe em versão preferida (validado > provisório > beta, CRID mais novo)"
    rows.sort(key=lambda m: (m.get("inicio") or "", m["produto"]))
    return rows, objs, cols


def funnel_table(rows):
    by = defaultdict(lambda: [0, 0, 0])
    for m in rows:
        p = m["produto"]
        by[p][0] += 1
        by[p][1] += m["etapa_final"] in ("contem_ponto", "uma_por_passagem")
        by[p][2] += m["etapa_final"] == "uma_por_passagem"
    return {p: {"catalogo": v[0], "contem_ponto": v[1], "uma_por_passagem": v[2]} for p, v in sorted(by.items())}


def disk_ring(grid, lon, lat, r_in, r_out=None):
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, grid.crs, always_xy=True).transform(lon, lat)
    cols = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
    rows = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
    X, Y = np.meshgrid(cols, rows)
    d = np.hypot(X - x, Y - y)
    return (d <= r_in) if r_out is None else ((d > r_in) & (d <= r_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True); ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--nome", default="Ponto"); ap.add_argument("--out", required=True)
    ap.add_argument("--inicio", default=LAUNCH); ap.add_argument("--fim", default=date.today().isoformat())
    ap.add_argument("--raio-m", type=float, default=500.0)
    ap.add_argument("--sem-leitura", action="store_true", help="só o inventário de metadados")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "miniaturas").mkdir(exist_ok=True)
    cache = Path("cache"); cache.mkdir(exist_ok=True)
    from deltal import sources_nisar as sn
    from deltal import export
    sn.login()
    lon, lat = a.lon, a.lat
    ficha = {"nome": a.nome, "ponto": {"lat": lat, "lon": lon}, "periodo": [a.inicio, a.fim], "raio_m": a.raio_m,
             "gerado_em": datetime.now().isoformat(timespec="seconds"), "metodo": __doc__.strip(), "erros": {}}

    # 1) inventário
    log("inventário de todos os produtos NISAR que contêm o ponto ...")
    rows, objs, cols = inventory_point(lon, lat, a.inicio, a.fim)
    keys = sorted({k for r in rows for k in r})
    with open(out / "metadados_ponto.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    ficha["colecoes"] = cols
    ficha["funil_por_produto"] = funnel_table(rows)
    kept = [r for r in rows if r["etapa_final"] == "uma_por_passagem"]
    geo_count = defaultdict(int)
    for r in kept:
        if r["produto"] == "GCOV":
            geo_count[_geo_key(r)] += 1
    ficha["geometrias_gcov"] = dict(sorted(geo_count.items()))
    from deltal.inventory import calendar_png
    calendar_png(out / "calendario_produtos.png", rows, a.inicio, a.fim, GAPS, title=f"{a.nome}: produtos NISAR que contêm o ponto")
    log("funil por produto: " + json.dumps(ficha["funil_por_produto"], ensure_ascii=False))
    if a.sem_leitura:
        report.write_ficha(out, ficha); return

    # 2) GCOV no ponto
    grid = make_grid([lon - 0.015, lat - 0.015, lon + 0.015, lat + 0.015], 20)      # ~3 km
    disk = disk_ring(grid, lon, lat, a.raio_m)
    ring = disk_ring(grid, lon, lat, 1000, 1500)
    gcov = [r for r in kept if r["produto"] == "GCOV"]
    pts = []
    for i, r in enumerate(gcov, 1):
        snm, g = objs[r["arquivo"]]
        f = cache / f"{r['arquivo']}_pt2_{lat:.4f}_{lon:.4f}.npz"
        t0 = time.time()
        log(f"GCOV {i}/{len(gcov)} {r.get('inicio', '')[:8]} {_geo_key(r)} {r['maturidade']}")
        try:
            if f.exists():
                z = np.load(f, allow_pickle=False); arrs = {k: z[k] for k in z.files if k != "meta"}; meta = json.loads(str(z["meta"]))
            else:
                arrs, meta = sn.read_gcov_subset(g, grid)
                np.savez_compressed(f, meta=json.dumps(meta, default=str), **{k: v.astype("float32") for k, v in arrs.items()})
        except Exception as e:
            ficha["erros"][r["arquivo"]] = f"{type(e).__name__}: {e}"; log(f"   falhou: {e}"); continue
        hh = arrs["HHHH"]
        hv = arrs.get("HVHV", np.full(hh.shape, np.nan))
        okm = disk & np.isfinite(hh)
        relief = sn.relief_mask(arrs["mascara_relevo"]) if "mascara_relevo" in arrs else np.zeros(hh.shape, bool)
        dt = datetime.strptime(r["inicio"], "%Y%m%dT%H%M%S")
        row = {"data": dt.strftime("%Y-%m-%d"), "hora_utc": dt.strftime("%H:%M"), "geometria": _geo_key(r),
               "maturidade": r["maturidade"], "crid": r.get("crid"), "modo_banda": r.get("modo_banda"), "polarizacao": r.get("polarizacao"),
               "pixels_validos": int(okm.sum()), "relevo_mascarado_pct": round(100 * float(relief[disk].mean()), 1),
               "hh_db": round(float(to_db(np.nanmean(hh[okm]))), 3) if okm.any() else None,
               "hv_db": round(float(to_db(np.nanmean(hv[okm]))), 3) if np.isfinite(hv[okm]).any() else None,
               "hv_anel_db": round(float(to_db(np.nanmean(hv[ring]))), 3) if np.isfinite(hv[ring]).any() else None,
               "hh_anel_db": round(float(to_db(np.nanmean(hh[ring]))), 3) if np.isfinite(hh[ring]).any() else None,
               "arquivo": r["arquivo"], "segundos": round(time.time() - t0, 1)}
        idt = meta.get("identificacao") or {}
        for k in ("lookDirection", "orbitPassDirection", "absoluteOrbitNumber", "isUrgentObservation", "isJointObservation"):
            if k in idt:
                row["id_" + k] = idt[k]
        if row["hh_db"] is not None and row["hv_db"] is not None:
            row["hh_menos_hv_db"] = round(row["hh_db"] - row["hv_db"], 3)
        if row["hh_db"] is not None and row["hh_anel_db"] is not None:
            row["hh_contraste_db"] = round(row["hh_db"] - row["hh_anel_db"], 3)     # ponto menos o entorno
        if row["hv_db"] is not None and row["hv_anel_db"] is not None:
            row["hv_contraste_db"] = round(row["hv_db"] - row["hv_anel_db"], 3)
        row["mascaras"] = json.dumps(meta.get("valores_mascaras", {}))
        pts.append(row)
        # miniatura (HV em dB, 3 km) para a animação do ponto
        try:
            v = to_db(hv)
            plt.imsave(out / "miniaturas" / f"{row['data']}_{row['geometria']}.png", np.nan_to_num(np.clip((v + 30) / 25, 0, 1)), cmap="gray")
        except Exception:
            pass
    pts.sort(key=lambda x: x["data"])

    # 3) clima
    clima = {}
    try:
        from deltal.sources_power import climate_daily, antecedent_index
        clima = climate_daily(lon, lat, (datetime.fromisoformat(a.inicio) - timedelta(days=10)).strftime("%Y%m%d"), a.fim.replace("-", ""))
        for p in pts:
            p["chuva_7d_mm"] = round(antecedent_index(clima.get("PRECTOTCORR", {}), p["data"].replace("-", "")), 1)
            p["umidade_solo_sup"] = clima.get("GWETTOP", {}).get((datetime.fromisoformat(p["data"]) - timedelta(days=1)).strftime("%Y%m%d"))
    except Exception as e:
        ficha["erros"]["clima"] = f"{type(e).__name__}: {e}"

    # 3b) SME2: umidade do solo do próprio NISAR no ponto
    sme = [r for r in kept if r["produto"] == "SME2"]
    sme_rows = []
    for i, r in enumerate(sme, 1):
        snm, g = objs[r["arquivo"]]
        log(f"SME2 {i}/{len(sme)} {r.get('inicio', '')[:8]} {_geo_key(r)}")
        try:
            v = sn.read_sme2_point(g, lon, lat)
        except Exception as e:
            v = {"erro": f"{type(e).__name__}: {e}"}
        v.update({"data": f"{r['inicio'][:4]}-{r['inicio'][4:6]}-{r['inicio'][6:8]}", "geometria": _geo_key(r), "arquivo": r["arquivo"]})
        sme_rows.append(v)
        if "erro" in v and i == 1:
            log(f"   SME2 não lido: {v['erro']} (estrutura registrada na ficha)")
    ficha["sme2"] = sme_rows
    sm_by = {(x["data"], x["geometria"]): x.get("valor") for x in sme_rows}
    for p in pts:
        p["nisar_umidade"] = sm_by.get((p["data"], p["geometria"]))

    # 4) GUNW no ponto
    gunw_rows = []
    gun = [r for r in kept if r["produto"] == "GUNW"]
    for i, r in enumerate(gun, 1):
        snm, g = objs[r["arquivo"]]
        f = cache / f"{r['arquivo']}_pt2_{lat:.4f}_{lon:.4f}.npz"
        log(f"GUNW {i}/{len(gun)} {r.get('inicio', '')[:8]} -> {(r.get('inicio_par2') or '')[:8]} {_geo_key(r)}")
        try:
            if f.exists():
                z = np.load(f); arrs = {k: z[k] for k in z.files}
            else:
                arrs, _ = sn.read_gunw_subset(g, grid)
                np.savez_compressed(f, **{k: v.astype("float32") for k, v in arrs.items()})
        except Exception as e:
            ficha["erros"][r["arquivo"]] = f"{type(e).__name__}: {e}"; log(f"   falhou: {e}"); continue
        d, c = arrs["deslocamento_cm"], arrs["coerencia"]
        gunw_rows.append({"data1": f"{r['inicio'][:4]}-{r['inicio'][4:6]}-{r['inicio'][6:8]}",
                          "data2": f"{r['inicio_par2'][:4]}-{r['inicio_par2'][4:6]}-{r['inicio_par2'][6:8]}" if r.get("inicio_par2") else None,
                          "geometria": _geo_key(r), "maturidade": r["maturidade"],
                          "coerencia_ponto": round(float(np.nanmean(c[disk])), 3) if np.isfinite(c[disk]).any() else None,
                          "coerencia_anel": round(float(np.nanmean(c[ring])), 3) if np.isfinite(c[ring]).any() else None,
                          "deslocamento_relativo_cm": round(float(np.nanmedian(d[disk]) - np.nanmedian(d[ring])), 2)
                          if np.isfinite(d[disk]).any() and np.isfinite(d[ring]).any() else None,
                          "arquivo": r["arquivo"]})

    # 5) passagens anômalas: o ENTORNO inteiro (anel de 1 a 1,5 km) saltou mais de 3 dB em relação
    #    à mediana da geometria. Mudança real no chão não muda tudo ao redor de uma vez; isso é a passagem.
    by_geo = defaultdict(list)
    for p in pts:
        if p["hh_db"] is not None:
            by_geo[p["geometria"]].append(p)
    anomalas = []
    for gk, ps in by_geo.items():
        med = np.nanmedian([p["hh_anel_db"] for p in ps if p["hh_anel_db"] is not None])
        for p in ps:
            p["passagem_anomala"] = bool(p["hh_anel_db"] is not None and abs(p["hh_anel_db"] - med) > 3.0)
            if p["passagem_anomala"]:
                anomalas.append({"data": p["data"], "geometria": gk, "salto_do_entorno_db": round(p["hh_anel_db"] - med, 2)})
    ficha["passagens_anomalas"] = anomalas
    by_geo = {gk: [p for p in ps if not p["passagem_anomala"]] for gk, ps in by_geo.items()}
    ana = {}
    for gk, ps in by_geo.items():
        if len(ps) < 4:
            ana[gk] = {"passagens": len(ps), "nota": "menos de 4 passagens: sem análise de quebra"}
            continue
        dates = [p["data"] for p in ps]
        nan = float("nan")
        hv = np.array([[[p["hv_db"] if p["hv_db"] is not None else nan]] for p in ps]); hh = np.array([[[p["hh_db"]]] for p in ps])
        t_days = [(datetime.fromisoformat(x) - datetime.fromisoformat(dates[0])).days for x in dates]
        base = hv if np.isfinite(hv).sum() >= 4 else hh          # só HH nas pistas de polarização simples
        k, dlt, tt, resid = S.breakpoint(base, 2)
        slope, ts = S.trend(base, t_days)
        e = {"passagens": len(ps), "primeira": dates[0], "ultima": dates[-1],
             "passagens_com_hv": int(np.isfinite(hv).sum()),
             "hv_media_db": round(float(np.nanmean(hv)), 2), "hv_amplitude_db": round(float(np.nanmax(hv) - np.nanmin(hv)), 2),
             "hh_amplitude_db": round(float(np.nanmax(hh) - np.nanmin(hh)), 2),
             "canal_da_analise": "HV" if base is hv else "HH",
             "maior_degrau_db": round(float(dlt[0, 0]), 2), "t_do_degrau": round(float(tt[0, 0]), 1),
             "degrau_entre": [dates[int(k[0, 0]) - 1], dates[int(k[0, 0])]] if k[0, 0] > 0 else None,
             "tendencia_db_mes": round(float(slope[0, 0]), 3), "t_tendencia": round(float(ts[0, 0]), 1)}
        if clima:
            w = [p.get("chuva_7d_mm") for p in ps]
            r_hh, _ = S.moisture_response(hh, w); r_hv, _ = S.moisture_response(hv, w)
            e["r_hh_chuva"], e["r_hv_chuva"] = round(float(r_hh[0, 0]), 3), round(float(r_hv[0, 0]), 3)
            u = [p.get("umidade_solo_sup") if p.get("umidade_solo_sup") is not None else nan for p in ps]
            r_u, sl_u = S.moisture_response(hh, u)
            e["r_hh_umidade_solo"], e["hh_db_por_0_1_umidade"] = round(float(r_u[0, 0]), 3), round(float(sl_u[0, 0]) * 0.1, 2)
            if "nisar_umidade" in ps[0] or any(p.get("nisar_umidade") is not None for p in ps):
                n_ = [p.get("nisar_umidade") if p.get("nisar_umidade") is not None else nan for p in ps]
                r_n, _ = S.moisture_response(hh, n_)
                e["r_hh_umidade_nisar_sme2"] = round(float(r_n[0, 0]), 3)
        cst = np.array([[[p.get("hh_contraste_db", nan)]] for p in ps])
        sc, tc = S.trend(cst, t_days)
        e["contraste_hh_tendencia_db_mes"], e["t_contraste"] = round(float(sc[0, 0]), 3), round(float(tc[0, 0]), 1)
        ana[gk] = e
    ficha["analise_por_geometria"] = ana
    ficha["gunw"] = gunw_rows

    # saídas
    if pts:
        with open(out / "serie_ponto.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for p in pts for k in p})); w.writeheader(); w.writerows(pts)
    if gunw_rows:
        with open(out / "gunw_ponto.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(gunw_rows[0])); w.writeheader(); w.writerows(gunw_rows)
    plot_point(out / "historia_do_ponto.png", a.nome, by_geo, gunw_rows, clima, a.inicio, a.fim, anomalas, sme_rows)
    try:
        from PIL import Image, ImageDraw
        frames = []
        for p in pts:
            fp = out / "miniaturas" / f"{p['data']}_{p['geometria']}.png"
            if fp.exists():
                im = Image.open(fp).convert("RGB").resize((300, 300))
                dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 300, 16], fill=(7, 23, 63))
                dr.text((4, 2), f"{p['data']} {p['geometria']} HV", fill=(234, 254, 7)); dr.ellipse([145, 145, 155, 155], outline=(255, 107, 0))
                frames.append(im)
        if frames:
            frames[0].save(out / "animacao_ponto.gif", save_all=True, append_images=frames[1:], duration=500, loop=0)
    except Exception as e:
        ficha["erros"]["animacao"] = str(e)
    ficha["serie"] = pts
    fic = report.write_ficha(out, ficha)
    lines = [f"# {a.nome} ({lat}, {lon}): a história no NISAR", "", f"Período {a.inicio} a {a.fim}.", "",
             "| Produto | catálogo | contém o ponto | uma por passagem |", "|---|---|---|---|"]
    lines += [f"| {p} | {v['catalogo']} | {v['contem_ponto']} | {v['uma_por_passagem']} |" for p, v in fic["funil_por_produto"].items()]
    if fic.get("passagens_anomalas"):
        lines += ["", "## Passagens anômalas (fora da análise)", ""] + [f"- {x['data']} {x['geometria']}: o entorno inteiro saltou {x['salto_do_entorno_db']} dB" for x in fic["passagens_anomalas"]]
    if sme_rows:
        ok_s = [x for x in sme_rows if x.get("valor") is not None]
        lines += ["", f"## SME2 (umidade do solo NISAR): {len(ok_s)} de {len(sme_rows)} lidos", ""]
        lines += [f"- {x['data']} {x['geometria']}: {round(x['valor'], 3)}" for x in ok_s]
        if not ok_s:
            lines.append(f"- não lido: {sme_rows[0].get('erro')}")
    lines += ["", "## Por geometria (GCOV)", ""]
    for gk, e in fic["analise_por_geometria"].items():
        lines.append(f"- **{gk}**: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in e.items()))
    if gunw_rows:
        lines += ["", f"## GUNW: {len(gunw_rows)} pares", ""] + [f"- {g['data1']} → {g['data2']} ({g['geometria']}): coerência {g['coerencia_ponto']}, "
                                                               f"deslocamento relativo {g['deslocamento_relativo_cm']} cm" for g in gunw_rows]
    if fic["erros"]:
        lines += ["", "## Erros", ""] + [f"- {k}: {v}" for k, v in fic["erros"].items()]
    (out / "resultado.md").write_text("\n".join(lines), encoding="utf-8")
    log("pronto")
    print("\n".join(lines))


def plot_point(path, nome, by_geo, gunw_rows, clima, start, end, anomalas=(), sme_rows=()):
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=110, sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    cols = [report.ELECTRIC, "#FF6B00", "#3BB273", "#B5179E", report.YONDER, "#7A5C00"]
    for i, (gk, ps) in enumerate(sorted(by_geo.items())):
        d = [datetime.fromisoformat(p["data"]) for p in ps]
        c = cols[i % len(cols)]
        axs[0].plot(d, [p["hv_db"] if p["hv_db"] is not None else np.nan for p in ps], "o-", color=c, label=f"HV {gk} ({len(ps)})")
        axs[0].plot(d, [p["hh_db"] for p in ps], "s--", ms=3, lw=0.8, color=c, alpha=0.6, label=f"HH {gk}")
    axs[0].set_ylabel("radar dB"); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title(f"{nome}: o ponto ao longo do tempo", color=report.DEEP)
    rain = clima.get("PRECTOTCORR") if clima else None
    if rain:
        axs[1].bar([datetime.strptime(k, "%Y%m%d") for k in rain], list(rain.values()), width=1, color="#7FB3D5")
        axs[1].set_ylabel("chuva mm/dia")
        if clima.get("GWETTOP"):
            t = axs[1].twinx(); t.plot([datetime.strptime(k, "%Y%m%d") for k in clima["GWETTOP"]], list(clima["GWETTOP"].values()), color="#0B6E4F")
            t.set_ylim(0, 1); t.set_ylabel("umidade do solo")
            sv = [(datetime.fromisoformat(x["data"]), x["valor"]) for x in sme_rows if x.get("valor") is not None]
            if sv:
                t.plot(*zip(*sv), "D", color="#B5179E", ms=5, label="NISAR SME2"); t.legend(fontsize=7, loc="upper right")
    for g in gunw_rows:
        if g["coerencia_ponto"] is not None and g["data2"]:
            axs[2].plot([datetime.fromisoformat(g["data1"]), datetime.fromisoformat(g["data2"])], [g["coerencia_ponto"]] * 2, "-", lw=2, color=report.ELECTRIC)
    axs[2].set_ylim(0, 1); axs[2].set_ylabel("coerência GUNW")
    for ax in axs:
        ax.axvspan(datetime.fromisoformat(LAUNCH), datetime.fromisoformat(BETA_END), color="#eee", alpha=0.7)
        for g0, g1, _ in GAPS:
            ax.axvspan(datetime.fromisoformat(g0), datetime.fromisoformat(g1), color="#ccc", alpha=0.5)
    for an in anomalas:
        axs[0].axvline(datetime.fromisoformat(an["data"]), color="red", ls=":", lw=1)
        axs[0].text(datetime.fromisoformat(an["data"]), axs[0].get_ylim()[0], f" passagem anômala {an['geometria']}", rotation=90,
                    va="bottom", fontsize=7, color="red")
    axs[0].text(datetime.fromisoformat(LAUNCH), axs[0].get_ylim()[1], " beta", va="top", fontsize=8, color="#666")
    axs[2].set_xlim(datetime.fromisoformat(start), datetime.fromisoformat(end))
    fig.autofmt_xdate(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
