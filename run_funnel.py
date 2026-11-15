"""Busca unificada em funil + consenso multissensor + validação.

Uso:
    python run_funnel.py --case cases/norte_fire.json --out docs/results/norte_fire_funil [--skip-nisar] [--skip-naip]

Saídas: funil.json (contagem e aproveitamento por etapa e fonte), ficha_tecnica.json/.csv,
resultado.md, figuras PNG/JPG e docs/api/v1/analyses/<nome>.json (a "API GET" estática).
"""
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deltal import __version__, collect, report
from deltal.change import nbr, min_area_filter, to_db
from deltal.consensus import combine
from deltal.grid import make_grid
from deltal.layers import compute_layers
from deltal.terrain import radar_masks, slope_deg
from deltal.validate import metrics

TRUTH = {"dnbr_min": 0.1, "min_area_m2": 10000.0}
CAL_C = {"veg_drop_db": -6.0, "min_area_m2": 40000.0}     # calibrado no caso de treino (Incêndio Norte)
LANDSAT = {"dnbr_min": 0.1, "dlst_min_c": 3.0}
CONSENSUS = {"min_votes": 2, "min_area_m2": 40000.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-nisar", action="store_true")
    ap.add_argument("--skip-naip", action="store_true")
    a = ap.parse_args()

    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(case["bbox"], case.get("resolucao_m", 20))
    ev = case["evento"]["data"]
    px = grid.pixel_area_m2
    ficha = {"nome": case["nome"], "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "versao_pipeline": __version__, "area_bbox": case["bbox"], "evento": case["evento"],
             "grade": {"crs": grid.crs, "resolucao_m": grid.resolution, "largura": grid.width, "altura": grid.height},
             "parametros": {"verdade": TRUTH, "banda_C": CAL_C, "landsat": LANDSAT, "consenso": CONSENSUS},
             "funil": [], "fontes": {}, "resultados": {}, "contexto": {}, "erros": {}}
    st = {}

    def step(name, fn):
        try:
            return fn()
        except Exception as e:
            ficha["erros"][name] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            return None

    # ---------------- Funis -------------------------------------------------------
    def f_s2():
        fun, best = collect.sentinel2(grid, ev); ficha["funil"].append(fun.summary()); st["s2"] = best
    def f_ls():
        fun, best = collect.landsat(grid, ev); ficha["funil"].append(fun.summary()); st["ls"] = best
    def f_s1():
        fun, best = collect.sentinel1(grid, ev); ficha["funil"].append(fun.summary()); st["s1"] = best
    def f_naip():
        fun, best = collect.naip(grid, ev); ficha["funil"].append(fun.summary()); st["naip"] = best
    def f_aster():
        fun, best = collect.aster(grid, ev); ficha["funil"].append(fun.summary())
    def f_dem():
        fun, z = collect.dem(grid); ficha["funil"].append(fun.summary()); st["dem"] = z
    def f_nisar():
        from deltal import sources_nisar as sn
        sn.login()
        fun, best = sn.funnel(grid, ev); ficha["funil"].append(fun.summary()); st["nisar"] = best
    for name, fn in [("funil_sentinel2", f_s2), ("funil_landsat", f_ls), ("funil_sentinel1", f_s1),
                     ("funil_aster", f_aster), ("funil_dem", f_dem)]:
        step(name, fn)
    if not a.skip_naip:
        step("funil_naip", f_naip)
    if not a.skip_nisar:
        step("funil_nisar", f_nisar)

    # ---------------- Verdade de campo (Sentinel-2) ---------------------------------
    truth = valid_t = land = None
    def s2():
        _, pre, post, gap = st["s2"]
        r0, c0, w0 = collect.sentinel2_load(pre, grid)
        r1, c1, w1 = collect.sentinel2_load(post, grid)
        with np.errstate(invalid="ignore"):
            dnbr = nbr(r0["B08"], r0["B12"]) - nbr(r1["B08"], r1["B12"])
            land_ = c0 & c1 & ~w0 & ~w1
            v = land_ & np.isfinite(dnbr)
            t = v & (dnbr > TRUTH["dnbr_min"])
        t, sizes = min_area_filter(t, int(TRUTH["min_area_m2"] / px))
        ficha["fontes"]["sentinel2"] = {"antes": pre.day, "depois": post.day, "itens_antes": pre.ids, "itens_depois": post.ids}
        res = {"area_queimada_ha": t.sum() * px / 1e4, "maior_grupo_ha": (max(sizes) if sizes else 0) * px / 1e4}
        if case["evento"].get("area_oficial_ha"):
            res["area_oficial_ha"] = case["evento"]["area_oficial_ha"]
            res["razao_maior_grupo_oficial"] = res["maior_grupo_ha"] / case["evento"]["area_oficial_ha"]
        ficha["resultados"]["verdade_sentinel2"] = res
        rgb = lambda r: np.nan_to_num(np.dstack([report._stretch(r["B04"]), report._stretch(r["B03"]), report._stretch(r["B02"])]))
        report.overlay_png(out / "s2_antes_cor_natural.png", rgb(r0), title=f"Sentinel-2 antes ({pre.day})")
        report.overlay_png(out / "s2_depois_cor_natural.png", rgb(r1), truth=t, title=f"Sentinel-2 depois ({post.day}), cicatriz em branco")
        return t, v, land_
    r = step("verdade_sentinel2", s2)
    if r:
        truth, valid_t, land = r

    # ---------------- Relevo ------------------------------------------------------
    def terrain_for(sensor, orbit):
        if st.get("dem") is None:
            return None
        lay, sha, _ = radar_masks(st["dem"], grid.resolution, sensor, orbit)
        return lay | sha
    if st.get("dem") is not None:
        ficha["resultados"]["relevo"] = {"declividade_media_graus": float(np.nanmean(slope_deg(st["dem"], grid.resolution)))}

    evid = {}

    # ---------------- Sentinel-1 --------------------------------------------------
    def s1():
        key, pre, post, gap = st["s1"]
        b0, b1 = collect.sentinel1_load(pre, grid), collect.sentinel1_load(post, grid)
        bad = terrain_for("sentinel-1", key[1])
        v = np.isfinite(b0["vv"]) & np.isfinite(b1["vv"])
        if land is not None:
            v &= land
        if bad is not None:
            v &= ~bad
        res = compute_layers(b0["vv"], b1["vv"], b0["vh"], b1["vh"], px, valid=v, params=CAL_C)
        e = {"stats": res["stats"], "intervalo_dias": gap, "orbita": {"relativa": key[0], "direcao": key[1]},
             "pixels_mascarados_relevo": int(bad.sum()) if bad is not None else None}
        if truth is not None:
            e["metricas"] = metrics(res["vegetacao"], truth, valid_t & v, px)
        ficha["fontes"]["sentinel1"] = {"antes": pre.ids, "depois": post.ids}
        ficha["resultados"]["sentinel1"] = e
        evid["sentinel1"] = (res["vegetacao"], v)
        report.save_png(out / "s1_mudanca_vh.png", res["lr_cross"], title="Sentinel-1 VH, mudança (dB)")
    if st.get("s1"):
        step("sentinel1", s1)

    # ---------------- Landsat (óptico + térmico) ----------------------------------
    def ls():
        _, pre, post, gap = st["ls"]
        s0, t0, c0, w0 = collect.landsat_load(pre, grid)
        s1_, t1, c1, w1 = collect.landsat_load(post, grid)
        v = c0 & c1 & ~w0 & ~w1 & np.isfinite(t0) & np.isfinite(t1)
        with np.errstate(invalid="ignore"):
            dnbr_l = nbr(s0["nir08"], s0["swir22"]) - nbr(s1_["nir08"], s1_["swir22"])
            # temperatura relativa: cada data menos a mediana da terra (tira o efeito do dia mais quente)
            rel0 = t0 - np.nanmedian(t0[v]); rel1 = t1 - np.nanmedian(t1[v])
            dlst = rel1 - rel0
        burn, _ = min_area_filter(v & (dnbr_l > LANDSAT["dnbr_min"]), int(TRUTH["min_area_m2"] / px))
        hot, _ = min_area_filter(v & (dlst > LANDSAT["dlst_min_c"]), int(TRUTH["min_area_m2"] / px))
        e = {"intervalo_dias": gap, "lst_mediana_antes_c": float(np.nanmedian(t0[v])), "lst_mediana_depois_c": float(np.nanmedian(t1[v])),
             "area_dnbr_ha": burn.sum() * px / 1e4, "area_aquecida_ha": hot.sum() * px / 1e4}
        if truth is not None:
            e["metricas_dnbr"] = metrics(burn, truth, valid_t & v, px)
            e["metricas_termico"] = metrics(hot, truth, valid_t & v, px)
            e["aquecimento_na_cicatriz_c"] = float(np.nanmedian(dlst[truth & v])) if np.any(truth & v) else None
        ficha["fontes"]["landsat"] = {"antes": pre.ids, "depois": post.ids}
        ficha["resultados"]["landsat"] = e
        evid["landsat_dnbr"] = (burn, v)
        evid["landsat_termico"] = (hot, v)
        report.save_png(out / "landsat_termico_delta.png", dlst, cmap="inferno", vmin=-2, vmax=10,
                        title=f"Landsat: aquecimento relativo da superfície (°C) {pre.day} → {post.day}")
    if st.get("ls"):
        step("landsat", ls)

    # ---------------- NISAR -------------------------------------------------------
    def nis():
        from deltal import sources_nisar as sn
        key, (s0, g0), (s1n, g1), gap = st["nisar"]
        (a0, m0), (a1, m1) = sn.read_gcov_subset(g0, grid), sn.read_gcov_subset(g1, grid)
        bad = terrain_for("nisar", m0["direcao"])
        v = np.isfinite(a0["HHHH"]) & np.isfinite(a1["HHHH"])
        if land is not None:
            v &= land
        if bad is not None:
            v &= ~bad
        res = compute_layers(a0["HHHH"], a1["HHHH"], a0["HVHV"], a1["HVHV"], px, valid=v)
        e = {"stats": res["stats"], "intervalo_dias": gap, "colecoes": [s0, s1n]}
        if truth is not None:
            e["metricas"] = metrics(res["vegetacao"], truth, valid_t & v, px)
        ficha["fontes"]["nisar"] = {"antes": m0, "depois": m1}
        ficha["resultados"]["nisar"] = e
        evid["nisar"] = (res["vegetacao"], v)
        report.save_png(out / "nisar_mudanca_hv.png", res["lr_cross"], title="NISAR HV, mudança (dB)")
    if st.get("nisar"):
        step("nisar", nis)

    # ---------------- Consenso (sem o Sentinel-2, que é a verdade) ----------------
    def cons():
        c = combine(evid, CONSENSUS["min_votes"], px, CONSENSUS["min_area_m2"])
        e = {"fontes": c["fontes"], "area_consenso_ha": c["consenso"].sum() * px / 1e4}
        if truth is not None:
            e["metricas"] = metrics(c["consenso"], truth, valid_t & (c["disponiveis"] >= CONSENSUS["min_votes"]), px)
            e["fontes_disponiveis_media"] = float(c["disponiveis"][valid_t].mean())
        ficha["resultados"]["consenso"] = e
        report.save_png(out / "confianca.png", c["confianca"], cmap="YlOrRd", vmin=0, vmax=1,
                        title="Confiança: fração das fontes que viram mudança")
        base = np.dstack([report._stretch(c["votos"].astype(float))] * 3)
        report.overlay_png(out / "consenso.png", np.nan_to_num(base), truth=truth,
                           layers={"vegetacao": c["consenso"]}, title="Consenso (laranja) contra a cicatriz Sentinel-2 (branco)")
    if len(evid) >= 2:
        step("consenso", cons)

    # ---------------- NAIP (antes em alta resolução) -------------------------------
    def nai():
        rgb, _ = collect.naip_load_rgb(st["naip"], grid)
        from PIL import Image
        Image.fromarray(rgb).save(out / "naip_antes.jpg", quality=85)
        ficha["fontes"]["naip"] = {"ano": st["naip"].dt.year, "itens": st["naip"].ids}
    if st.get("naip") and not a.skip_naip:
        step("naip", nai)

    # ---------------- Chuva -------------------------------------------------------
    def rain():
        from deltal.sources_power import rain_between
        w, s, e_, n = case["bbox"]
        d0 = (datetime.fromisoformat(ev).date().toordinal() - 12)
        d0 = datetime.fromordinal(d0).strftime("%Y%m%d")
        d1 = datetime.fromordinal(datetime.fromisoformat(ev).date().toordinal() + 12).strftime("%Y%m%d")
        ficha["contexto"]["chuva"] = rain_between((w + e_) / 2, (s + n) / 2, d0, d1)
    step("nasa_power", rain)

    fic = report.write_ficha(out, ficha)
    (out / "funil.json").write_text(json.dumps(fic["funil"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "resultado.md").write_text(report.narrative(fic), encoding="utf-8")
    api = Path("docs/api/v1/analyses"); api.mkdir(parents=True, exist_ok=True)
    name = Path(a.case).stem
    files = sorted(p.name for p in out.iterdir() if p.suffix in (".png", ".jpg", ".json", ".csv", ".md"))
    (api / f"{name}.json").write_text(json.dumps({"id": name, "status": "concluida", "ficha": fic,
                                                   "arquivos": [f"{out.as_posix()}/{f}" for f in files]},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"funil": fic["funil"], "resultados": fic["resultados"], "erros": fic["erros"]},
                     ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
