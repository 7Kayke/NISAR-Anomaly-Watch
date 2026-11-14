"""Roda um caso de validação completo.

Uso:
    python run_case.py --case cases/norte_fire.json --out results/norte_fire

Passos: grade comum -> verdade de campo (Sentinel-2 dNBR) -> Sentinel-1 RTC -> NISAR GCOV
-> chuva (NASA POWER) -> figuras, ficha técnica e texto. Cada passo que falhar fica
registrado na ficha, sem derrubar os outros.
"""
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deltal import __version__
from deltal.grid import make_grid
from deltal.change import nbr, ndvi, min_area_filter, to_db
from deltal.layers import compute_layers, PARAMS
from deltal.validate import metrics
from deltal import report

# Verdade de campo: dNBR > 0,1 (limite inferior de "queima de baixa severidade") sem filtro de NDVI,
# porque o capim seco de outubro também queima. Na 1ª rodada (dNBR > 0,27 + NDVI > 0,2) a cicatriz
# saiu com 121 ha contra ~231 ha do CAL FIRE; com dNBR > 0,1 o grupo principal dá ~227 ha.
TRUTH_PARAMS = {"dnbr_min": 0.1, "ndvi_pre_min": None, "min_area_m2": 10000.0}

# Calibração por banda. "C" (Sentinel-1) foi calibrada no caso de TREINO (Incêndio Norte) e congelada;
# as métricas desse caso são de treino, não de teste. "L" (NISAR) ainda não calibrada.
CALIBRATION = {
    "C": {"veg_drop_db": -6.0, "min_area_m2": 40000.0},
    "L": {},
}


def step(ficha, name, fn):
    try:
        return fn()
    except Exception as e:
        ficha.setdefault("erros", {})[name] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-nisar", action="store_true")
    a = ap.parse_args()

    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(case["bbox"], case.get("resolucao_m", 20))
    ficha = {
        "nome": case["nome"], "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_pipeline": __version__, "area_bbox": case["bbox"], "evento": case.get("evento"),
        "grade": {"crs": grid.crs, "resolucao_m": grid.resolution, "largura": grid.width, "altura": grid.height},
        "parametros_deteccao": PARAMS, "parametros_verdade": TRUTH_PARAMS,
        "fontes": {}, "resultados": {}, "contexto": {},
    }
    ev_date = case["evento"]["data"]
    arrays = {}

    # 1) Verdade de campo: Sentinel-2 dNBR -----------------------------------------
    def s2():
        from deltal.sources_pc import s2_pair, s2_load
        pre, post = s2_pair(grid, ev_date)
        r0, c0, w0, m0 = s2_load(pre, grid)
        r1, c1, w1, m1 = s2_load(post, grid)
        with np.errstate(invalid="ignore"):
            dnbr = nbr(r0["B08"], r0["B12"]) - nbr(r1["B08"], r1["B12"])
        if TRUTH_PARAMS["ndvi_pre_min"] is None:
            veg_pre = np.ones(dnbr.shape, bool)
        else:
            veg_pre = ndvi(r0["B08"], r0["B04"]) > TRUTH_PARAMS["ndvi_pre_min"]
        valid = c0 & c1 & ~w0 & ~w1 & np.isfinite(dnbr)
        with np.errstate(invalid="ignore"):
            burned = valid & veg_pre & (dnbr > TRUTH_PARAMS["dnbr_min"])
        burned, sizes = min_area_filter(burned, int(TRUTH_PARAMS["min_area_m2"] / grid.pixel_area_m2))
        ficha["fontes"]["sentinel2"] = {"antes": m0, "depois": m1, "colecao": "sentinel-2-l2a (Planetary Computer)"}
        area = burned.sum() * grid.pixel_area_m2 / 1e4
        entry = {"area_queimada_ha": area, "maior_grupo_ha": (max(sizes) if sizes else 0) * grid.pixel_area_m2 / 1e4,
                 "grupos": sizes, "pixels_validos": int(valid.sum())}
        oficial = case["evento"].get("area_oficial_ha")
        if oficial:
            entry["area_oficial_ha"] = oficial
            entry["razao_maior_grupo_oficial"] = entry["maior_grupo_ha"] / oficial
        ficha["resultados"]["verdade_sentinel2"] = entry
        report.save_png(out / "s2_dnbr.png", dnbr, cmap="RdYlGn_r", vmin=-0.5, vmax=0.8,
                        title=f"Sentinel-2 dNBR {m0['data']} → {m1['data']}")
        report.overlay_png(out / "s2_cicatriz.png", report.false_color(r1), truth=burned,
                           title="Cicatriz (contorno branco) sobre Sentinel-2 depois, falsa cor")
        land = c0 & c1 & ~w0 & ~w1
        arrays.update(dnbr=dnbr, truth=burned, truth_valid=valid, land=land)
        return burned, valid, land
    s2res = step(ficha, "sentinel2", s2)
    truth, truth_valid, land = s2res if s2res else (None, None, None)
    ficha["calibracao"] = {"banda_C": CALIBRATION["C"], "banda_L": CALIBRATION["L"] or "padrão (não calibrada)",
                           "caso_de_treino": "Incêndio Norte (2026-10-16)"}

    # 2) Sentinel-1 RTC -------------------------------------------------------------
    def s1():
        from deltal.sources_pc import s1_pair, s1_load
        gap, key, it0, it1 = s1_pair(grid, ev_date)
        b0, m0 = s1_load(it0, grid)
        b1, m1 = s1_load(it1, grid)
        v = None if land is None else (land & np.isfinite(b0["vv"]) & np.isfinite(b1["vv"]))
        res = compute_layers(b0["vv"], b1["vv"], b0["vh"], b1["vh"], grid.pixel_area_m2,
                             valid=v, params=CALIBRATION["C"])
        entry = {"stats": res["stats"], "intervalo_dias": gap, "orbita": list(key)}
        if truth is not None:
            v = truth_valid & np.isfinite(res["lr_cross"])
            entry["metricas"] = metrics(res["vegetacao"], truth, v, grid.pixel_area_m2)
        ficha["fontes"]["sentinel1_rtc"] = {"antes": m0, "depois": m1, "colecao": "sentinel-1-rtc (Planetary Computer)"}
        ficha["resultados"]["sentinel1_rtc"] = entry
        report.save_png(out / "s1_lr_vh.png", res["lr_cross"], title=f"Sentinel-1 VH, mudança (dB) {m0['data'][:10]} → {m1['data'][:10]}")
        base = np.dstack([report._stretch(to_db(b1["vh"]))] * 3)
        report.overlay_png(out / "s1_deteccao.png", np.nan_to_num(base), truth=truth,
                           layers={"vegetacao": res["vegetacao"], "agua": res["agua"]},
                           title="Sentinel-1: laranja = perda de vegetação, azul = água; branco = cicatriz S2")
        arrays.update(s1_lr_vh=res["lr_cross"], s1_veg=res["vegetacao"])
        return m0, m1
    s1dates = step(ficha, "sentinel1_rtc", s1)

    # 3) NISAR GCOV -----------------------------------------------------------------
    nis = case.get("nisar")
    if nis and not a.skip_nisar:
        def nisar():
            from deltal import sources_nisar as sn
            sn.login()
            grans = []
            for name, short in zip(nis["granulos"], nis["short_names"]):
                found = sn.search(None, None, None, short_name=short, granule_names=[name])
                if not found:
                    raise RuntimeError(f"Grânulo não encontrado no CMR: {name} ({short})")
                grans.append(found[0])
            (g0, m0), (g1, m1) = [sn.read_gcov_subset(g, grid) for g in grans]
            if (m0["pista"], m0["direcao"], m0["quadro"]) != (m1["pista"], m1["direcao"], m1["quadro"]):
                raise ValueError("Cenas NISAR de geometrias diferentes: não comparar")
            v = None if land is None else (land & np.isfinite(g0["HHHH"]) & np.isfinite(g1["HHHH"]))
            res = compute_layers(g0["HHHH"], g1["HHHH"], g0["HVHV"], g1["HVHV"], grid.pixel_area_m2,
                                 valid=v, params=CALIBRATION["L"])
            entry = {"stats": res["stats"]}
            if truth is not None:
                v = truth_valid & np.isfinite(res["lr_cross"])
                entry["metricas"] = metrics(res["vegetacao"], truth, v, grid.pixel_area_m2)
            ficha["fontes"]["nisar_gcov"] = {"antes": m0, "depois": m1}
            ficha["resultados"]["nisar_gcov"] = entry
            report.save_png(out / "nisar_lr_hv.png", res["lr_cross"], title="NISAR HV, mudança (dB)")
            base = np.dstack([report._stretch(to_db(g1["HVHV"]))] * 3)
            report.overlay_png(out / "nisar_deteccao.png", np.nan_to_num(base), truth=truth,
                               layers={"vegetacao": res["vegetacao"], "agua": res["agua"]},
                               title="NISAR: laranja = perda de vegetação, azul = água; branco = cicatriz S2")
            arrays.update(nisar_lr_hv=res["lr_cross"], nisar_veg=res["vegetacao"])
        step(ficha, "nisar_gcov", nisar)

    # 4) Chuva (NASA POWER) -----------------------------------------------------------
    def rain():
        from deltal.sources_power import rain_between
        w, s, e, n = case["bbox"]
        if s1dates:
            d0, d1 = s1dates[0]["data"][:10], s1dates[1]["data"][:10]
        else:
            from datetime import timedelta
            evd = datetime.fromisoformat(ev_date)
            d0, d1 = case["evento"].get("janela_chuva", [str((evd - timedelta(days=12)).date()),
                                                          str((evd + timedelta(days=12)).date())])
        ficha["contexto"]["chuva"] = rain_between((w + e) / 2, (s + n) / 2, d0.replace("-", ""), d1.replace("-", ""))
    step(ficha, "nasa_power", rain)

    np.savez_compressed(out / "arrays.npz", **{k: v for k, v in arrays.items() if v is not None})
    ficha = report.write_ficha(out, ficha)
    (out / "resultado.md").write_text(report.narrative(ficha), encoding="utf-8")
    print(json.dumps(ficha.get("resultados"), ensure_ascii=False, indent=2))
    print("erros:", ficha.get("erros"))


if __name__ == "__main__":
    main()
