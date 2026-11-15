# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Monta os dados do aplicativo (docs/app/data) a partir das saídas do pipeline.

Uso:
    python build_app_data.py --id pantanal --malha docs/results/scout_pantanal \
        --ponto docs/results/ponto_paiaguas [--ponto outra_pasta ...]

Modelo de dados (o contrato entre o pipeline e a interface):
  data/regioes.json                 índice: [{id, nome, bbox, periodo, pontos: [{id, nome, lat, lon, celula}]}]
  data/<regiao>/malha.geojson       quadros NISAR (tipo=quadro), cruzamentos A x D (tipo=celula), lugares (tipo=lugar)
  data/<regiao>/ponto_<id>.json     a história de um ponto: serie, gunw, clima, analise, achados (pt/en), limitacoes
Nenhum número é escrito à mão: tudo sai das fichas geradas pelo pipeline.
"""
import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

APP = Path("docs/app/data")


def slug(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _num(v):
    try:
        return None if v in (None, "", "None") else float(v)
    except (TypeError, ValueError):
        return None


def fmt(v, n=1):
    return f"{v:.{n}f}".replace(".", ",") if v is not None else "?"


def findings(f, serie, gunw):
    """Achados em linguagem simples, em pt e en, só a partir dos números."""
    out, lim = [], []
    ana = f.get("analise_por_geometria", {})
    # uma direção por trilho (quadros vizinhos do mesmo trilho são a mesma passagem)
    seen, uniq = set(), {}
    for gk in sorted(ana, key=lambda g: -(ana[g].get("passagens_com_hv") or 0)):
        tk = tuple(gk.split("_")[:2])
        if tk not in seen:
            seen.add(tk); uniq[gk] = ana[gk]
    ana = uniq
    # mesma queda vista por direções diferentes (quebra com t <= -2,5 e >= 0,5 dB, meio do intervalo a até 15 dias)
    drops = []
    for gk, e_ in ana.items():
        if e_.get("degrau_entre") and (e_.get("t_do_degrau") or 0) <= -2.5 and (e_.get("maior_degrau_db") or 0) <= -0.5:
            a_, b_ = [datetime.fromisoformat(x) for x in e_["degrau_entre"]]
            drops.append((a_ + (b_ - a_) / 2, gk, e_))
    drops.sort()
    best_cluster = []
    for i, (mid, gk, e_) in enumerate(drops):
        cl = [x for x in drops if abs((x[0] - mid).days) <= 15]
        if len(cl) > len(best_cluster):
            best_cluster = cl
    multi = best_cluster if len(best_cluster) >= 2 else []
    if multi:
        d0 = min(datetime.fromisoformat(x[2]["degrau_entre"][0]) for x in multi)
        d1 = max(datetime.fromisoformat(x[2]["degrau_entre"][1]) for x in multi)
        mags = [x[2]["maior_degrau_db"] for x in multi]
        lo_, hi_ = min(mags), max(mags)
        out.append({"tipo": "visivel", "valor": f"{fmt(hi_)} a {fmt(lo_)} dB",
                    "pt": f"A vegetação perdeu estrutura entre {d0:%d/%m} e {d1:%d/%m}: o HV caiu de {fmt(abs(hi_))} a {fmt(abs(lo_))} dB, "
                          f"e {len(multi)} direções de passagem independentes viram a mesma queda ({', '.join(x[1] for x in multi)}).",
                    "en": f"Vegetation lost structure between {d0:%b %d} and {d1:%b %d}: HV dropped by {fmt(abs(hi_))} to {fmt(abs(lo_))} dB, "
                          f"and {len(multi)} independent pass directions saw the same drop ({', '.join(x[1] for x in multi)})."})
    rs = [(gk, e_["r_hh_umidade_nisar_sme2"]) for gk, e_ in ana.items() if e_.get("r_hh_umidade_nisar_sme2") is not None]
    good_r = [r for _, r in rs if r >= 0.5]
    if good_r:
        out.append({"tipo": "invisivel", "valor": f"r {fmt(min(good_r), 2)}–{fmt(max(good_r), 2)}",
                    "pt": f"Água no solo (invisível ao olho): em {len(good_r)} de {len(rs)} direções de passagem, o brilho HH acompanha a umidade do solo medida pelo próprio NISAR (SME2), r de {fmt(min(good_r), 2)} a {fmt(max(good_r), 2)}.",
                    "en": f"Water in the soil (invisible to the eye): in {len(good_r)} of {len(rs)} pass directions, HH brightness follows the soil moisture measured by NISAR itself (SME2), r from {fmt(min(good_r), 2)} to {fmt(max(good_r), 2)}."})
    main = max(ana, key=lambda g: ana[g].get("passagens_com_hv", 0)) if ana else None
    e = ana.get(main, {}) if main else {}
    if e.get("passagens", 0) >= 4:
        hv = [s["hv"] for s in serie if s["g"] == main and s["hv"] is not None and not s["anom"]]
        strong = abs(e.get("t_do_degrau") or 0) >= 4 and abs(e.get("maior_degrau_db") or 0) >= 1
        if multi or good_r:
            pass
        elif strong and e.get("degrau_entre"):
            d0, d1 = e["degrau_entre"]
            up = (e["maior_degrau_db"] or 0) > 0
            out.append({"tipo": "visivel", "valor": f"{'+' if up else ''}{fmt(e['maior_degrau_db'])} dB",
                        "pt": f"Mudança na vegetação entre {d0} e {d1}: o retorno {e.get('canal_da_analise', 'HV')} "
                              f"{'subiu' if up else 'caiu'} {fmt(abs(e['maior_degrau_db']))} dB de uma vez.",
                        "en": f"Vegetation change between {d0} and {d1}: {e.get('canal_da_analise', 'HV')} backscatter "
                              f"{'rose' if up else 'dropped'} by {fmt(abs(e['maior_degrau_db']))} dB in one step."})
        elif hv:
            out.append({"tipo": "visivel", "valor": f"{fmt(max(hv) - min(hv))} dB",
                        "pt": f"A vegetação oscilou {fmt(max(hv) - min(hv))} dB no HV ao longo do período, sem um degrau forte.",
                        "en": f"Vegetation (HV) varied by {fmt(max(hv) - min(hv))} dB over the period, with no strong single step."})
        r = e.get("r_hh_chuva")
        if r is not None and not good_r:
            strength = "forte" if abs(r) >= 0.6 else ("moderada" if abs(r) >= 0.4 else "fraca")
            en_s = {"forte": "strong", "moderada": "moderate", "fraca": "weak"}[strength]
            txt_s = e.get("r_hh_umidade_nisar_sme2")
            out.append({"tipo": "invisivel", "valor": f"r {fmt(r, 2)}",
                        "pt": f"Água no solo (invisível ao olho): o HH acompanha a chuva dos 7 dias anteriores com relação {strength} "
                              f"(r = {fmt(r, 2)})" + (f"; com a umidade do solo do próprio NISAR, r = {fmt(txt_s, 2)}." if txt_s is not None else "."),
                        "en": f"Water in the soil (invisible to the eye): HH follows the previous 7 days of rain with a {en_s} relation "
                              f"(r = {fmt(r, 2)})" + (f"; with NISAR's own soil moisture, r = {fmt(txt_s, 2)}." if txt_s is not None else ".")})
        tc = e.get("t_contraste")
        if tc is not None and abs(tc) >= 2.5:
            sl = e.get("contraste_hh_tendencia_db_mes") or 0
            out.append({"tipo": "lento", "valor": f"{fmt(sl, 2)} dB/mês",
                        "pt": f"Mudança lenta: o ponto {'escurece' if sl < 0 else 'clareia'} em relação ao entorno, {fmt(abs(sl), 2)} dB por mês. É um indício, não uma conclusão.",
                        "en": f"Slow change: the point is getting {'darker' if sl < 0 else 'brighter'} than its surroundings, {fmt(abs(sl), 2)} dB per month. A hint, not a conclusion."})
    import statistics
    med = {}
    for g in gunw:
        if g["coh"] is not None:
            med.setdefault(g["g"], []).append(g["coh"])
    bad_geo = {k for k, v in med.items() if statistics.median(v) < 0.2}
    gunw = [g for g in gunw if g["g"] not in bad_geo]
    if bad_geo:
        lim.append({"pt": f"Direção(ões) {', '.join(sorted(bad_geo))} sem coerência útil neste ponto (mediana < 0,2): fora da análise do chão.",
                    "en": f"Pass direction(s) {', '.join(sorted(bad_geo))} have no useful coherence at this point (median < 0.2): left out of the ground analysis."})
    good = [g for g in gunw if g["coh"] is not None and g["coh"] >= 0.3 and g["desl"] is not None]
    if good:
        lo, hi = min(g["desl"] for g in good), max(g["desl"] for g in good)
        cohs = sorted(g["coh"] for g in good)
        out.append({"tipo": "chao", "valor": f"{fmt(lo)} a {fmt(hi)} cm",
                    "pt": f"O chão: em {len(good)} pares de interferometria confiáveis, o ponto se moveu entre {fmt(lo)} e {fmt(hi)} cm em relação ao entorno; coerência mediana {fmt(cohs[len(cohs) // 2], 2)}.",
                    "en": f"The ground: across {len(good)} reliable interferometric pairs, the point moved between {fmt(lo)} and {fmt(hi)} cm relative to its surroundings; median coherence {fmt(cohs[len(cohs) // 2], 2)}."})
    low_all = [g for g in gunw if g["coh"] is not None and g["coh"] < 0.5]
    mid_low = []                              # janela de 45 dias com mais trilhos diferentes em baixa coerência
    for g0 in low_all:
        a0 = datetime.fromisoformat(g0["d1"])
        win = [g for g in low_all if 0 <= (datetime.fromisoformat(g["d1"]) - a0).days <= 45]
        if len({g["g"].split("_")[0] for g in win}) > len({g["g"].split("_")[0] for g in mid_low}):
            mid_low = win
    tracks = {g["g"].split("_")[0] for g in mid_low}
    if len(tracks) >= 2:
        a_ = min(g["d1"] for g in mid_low); b_ = max(g["d2"] for g in mid_low)
        fa, fb = datetime.fromisoformat(a_), datetime.fromisoformat(b_)
        out.append({"tipo": "chao", "valor": f"{fa:%d/%m}–{fb:%d/%m}",
                    "pt": f"A superfície mexeu: a coerência caiu abaixo de 0,5 em {len(mid_low)} pares, vistos por {len(tracks)} trilhos diferentes, entre {fa:%d/%m} e {fb:%d/%m}. Algo alterou o chão (água baixando, queima, pisoteio).",
                    "en": f"The surface was disturbed: coherence fell below 0.5 in {len(mid_low)} pairs, seen from {len(tracks)} different tracks, between {fa:%b %d} and {fb:%b %d}. Something changed the ground (water receding, burning, trampling)."})
    low = [g for g in gunw if g["coh"] is not None and g["coh"] < 0.3]
    if low:
        out.append({"tipo": "chao", "valor": f"{len(low)} pares",
                    "pt": f"{len(low)} pares perderam a coerência (abaixo de 0,3): algo mexeu na superfície entre as datas, ou a passagem teve problema.",
                    "en": f"{len(low)} pairs lost coherence (below 0.3): something disturbed the surface between the dates, or the pass had a problem."})
    for an in f.get("passagens_anomalas", []):
        out.append({"tipo": "qualidade", "valor": an["data"],
                    "pt": f"Passagem de {an['data']} ({an['geometria']}) descartada: o entorno inteiro saltou {fmt(an['salto_do_entorno_db'])} dB de uma vez, sinal de defeito da aquisição.",
                    "en": f"Pass on {an['data']} ({an['geometria']}) discarded: the whole surrounding area jumped {fmt(an['salto_do_entorno_db'])} dB at once, a sign of an acquisition problem."})
    ds = sorted({s["d"] for s in serie})
    if len(ds) > 1:
        dd = [datetime.fromisoformat(x) for x in ds]
        gaps = sorted(((b - a).days, a, b) for a, b in zip(dd, dd[1:]))
        g, a, b = gaps[-1]
        if g > 24:
            lim.append({"pt": f"Sem passagens entre {a:%d/%m/%Y} e {b:%d/%m/%Y} ({g} dias).",
                        "en": f"No passes between {a:%Y-%m-%d} and {b:%Y-%m-%d} ({g} days)."})
        lim.append({"pt": f"Primeira passagem no ponto: {dd[0]:%d/%m/%Y}; última: {dd[-1]:%d/%m/%Y}.",
                    "en": f"First pass over the point: {dd[0]:%Y-%m-%d}; last: {dd[-1]:%Y-%m-%d}."})
    nsingle = sum(1 for s in serie if s["hv"] is None)
    if nsingle:
        lim.append({"pt": f"{nsingle} passagens só têm HH (polarização simples) e não mostram a vegetação (HV).",
                    "en": f"{nsingle} passes are HH-only (single polarization) and do not show vegetation (HV)."})
    lim.append({"pt": "Dados provisórios do NISAR: a calibração ainda pode mudar nas versões validadas.",
                "en": "Provisional NISAR data: calibration may still change in validated releases."})
    for x in out + lim:                       # inglês usa ponto decimal
        x["en"] = re.sub(r"(\d),(\d)", r"\1.\2", x["en"])
    for x in out:                             # o número do cartão nas duas línguas
        v = x["valor"]
        en = re.sub(r"(\d),(\d)", r"\1.\2", v).replace(" a ", " to ").replace("pares", "pairs").replace("mês", "month")
        x["valor"] = {"pt": v, "en": en}
    return out, lim


def build_point(folder, region_id):
    folder = Path(folder)
    f = json.loads((folder / "ficha_tecnica.json").read_text(encoding="utf-8"))
    pid = slug(f["nome"])
    serie = [{"d": s["data"], "h": s.get("hora_utc"), "g": s["geometria"], "m": s.get("maturidade"), "pol": s.get("polarizacao"),
              "hh": _num(s.get("hh_db")), "hv": _num(s.get("hv_db")), "hhc": _num(s.get("hh_contraste_db")),
              "anom": bool(s.get("passagem_anomala")), "chuva7": _num(s.get("chuva_7d_mm")), "umid": _num(s.get("umidade_solo_sup")),
              "sme2": _num(s.get("nisar_umidade"))} for s in f.get("serie", [])]
    gunw = [{"d1": g["data1"], "d2": g["data2"], "g": g["geometria"], "coh": _num(g.get("coerencia_ponto")),
             "desl": _num(g.get("deslocamento_relativo_cm"))} for g in f.get("gunw", [])]
    sme2 = [{"d": x["data"], "g": x["geometria"], "v": _num(x.get("valor"))} for x in f.get("sme2", []) if x.get("valor") is not None]
    clima = {}
    cf = folder / "clima.json"
    if cf.exists():
        c = json.loads(cf.read_text())
        iso = lambda k: f"{k[:4]}-{k[4:6]}-{k[6:8]}"
        clima = {"chuva_mm": {iso(k): v for k, v in c.get("PRECTOTCORR", {}).items()},
                 "umidade_solo": {iso(k): v for k, v in c.get("GWETTOP", {}).items()},
                 "temperatura_max_c": {iso(k): v for k, v in c.get("T2M_MAX", {}).items()}}
    ach, lim = findings(f, serie, gunw)
    out = {"id": pid, "regiao": region_id, "nome": f["nome"], "lat": f["ponto"]["lat"], "lon": f["ponto"]["lon"],
           "periodo": f["periodo"], "raio_m": f.get("raio_m"), "gerado_em": f.get("gerado_em"),
           "funil": f.get("funil_por_produto"), "geometrias": f.get("geometrias_gcov"),
           "serie": serie, "gunw": gunw, "sme2": sme2, "clima": clima, "analise": f.get("analise_por_geometria"),
           "achados": ach, "limitacoes": lim,
           "fonte": "NISAR L2 GCOV, L2 GUNW, L3 SME2 (NASA/JPL, ASF DAAC); clima: NASA POWER"}
    dest = APP / region_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"ponto_{pid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    for extra in ("animacao_ponto.gif", "historia_do_ponto.png"):
        if (folder / extra).exists():
            shutil.copy(folder / extra, dest / f"{pid}_{extra}")
    return {"id": pid, "nome": f["nome"], "lat": f["ponto"]["lat"], "lon": f["ponto"]["lon"], "arquivo": f"{region_id}/ponto_{pid}.json"}


def build_area(folder, region_id):
    """Mapas de mudança da área: uma entrada por geometria (pista/direção/quadro), mais concordância e coerência."""
    import numpy as np
    folder = Path(folder)
    dest = APP / region_id / "area"
    dest.mkdir(parents=True, exist_ok=True)
    geos = []
    for sub in sorted(p for p in folder.iterdir() if (p / "app" / "app.json").exists()):
        m = json.loads((sub / "app" / "app.json").read_text(encoding="utf-8"))
        gd = dest / sub.name
        gd.mkdir(exist_ok=True)
        for fn in ("app.json", "hv_u8.bin", "hh_u8.bin", "mapas_u8.bin"):
            shutil.copy(sub / "app" / fn, gd / fn)
        f = json.loads((sub / "ficha_tecnica.json").read_text(encoding="utf-8")) if (sub / "ficha_tecnica.json").exists() else {}
        res = f.get("resumo", {})
        geos.append({"id": sub.name, "pasta": f"{region_id}/area/{sub.name}", "passagens": len(m["datas"]),
                     "primeira": m["datas"][0], "ultima": m["datas"][-1], "forma": m["forma"],
                     "classes_ha": res.get("classes_ha"), "quedas_por_intervalo_ha": res.get("quedas_por_intervalo_ha"),
                     "valsa_invisivel": f.get("valsa_invisivel"), "relevo_mascarado_ha": f.get("relevo_mascarado_ha"),
                     "clima": f.get("clima", {}).get("chuva_antecedente_por_passagem_mm")})
    geos.sort(key=lambda g: -g["passagens"])
    out = {"geometrias": geos}
    if (folder / "assertividade.json").exists():
        out["assertividade"] = json.loads((folder / "assertividade.json").read_text(encoding="utf-8"))
    if (folder / "clima.json").exists():
        c = json.loads((folder / "clima.json").read_text())
        iso = lambda k: f"{k[:4]}-{k[4:6]}-{k[6:8]}"
        out["clima"] = {"chuva_mm": {iso(k): v for k, v in c.get("PRECTOTCORR", {}).items()},
                        "umidade_solo": {iso(k): v for k, v in c.get("GWETTOP", {}).items()}}
    gw = folder / "gunw"
    if (gw / "pares.json").exists():
        out["gunw"] = [p for p in json.loads((gw / "pares.json").read_text(encoding="utf-8")) if "erro" not in p]
        tif = gw / "coerencia_media.tif"
        if tif.exists() and geos:
            import rasterio
            with rasterio.open(tif) as r:
                c = r.read(1).astype("float64")
            _, H, W = geos[0]["forma"]
            st = max(1, c.shape[0] // H)
            c = c[:H * st, :W * st].reshape(H, st, W, st)
            with np.errstate(invalid="ignore"):
                c = np.nanmean(c, axis=(1, 3))
            np.where(np.isfinite(c), np.clip(c * 254 + 1, 1, 255), 0).astype("uint8").tofile(dest / "coerencia_u8.bin")
            out["coerencia"] = f"{region_id}/area/coerencia_u8.bin"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True); ap.add_argument("--malha", help="pasta do run_scout.py (com malha.geojson); opcional")
    ap.add_argument("--nome", help="nome da região, se não houver malha")
    ap.add_argument("--ponto", action="append", default=[])
    ap.add_argument("--caso", help="arquivo do caso (nome, contexto, ponto)")
    ap.add_argument("--serie", help="pasta do run_series.py da área (mapas de mudança por geometria)")
    a = ap.parse_args()
    dest = APP / a.id
    dest.mkdir(parents=True, exist_ok=True)
    g = {}
    if a.malha:
        g = json.loads((Path(a.malha) / "malha.geojson").read_text(encoding="utf-8"))
        (dest / "malha.geojson").write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    pts = [build_point(p, a.id) for p in a.ponto]
    area = build_area(a.serie, a.id) if a.serie else None
    idx_f = APP / "regioes.json"
    idx = json.loads(idx_f.read_text(encoding="utf-8")) if idx_f.exists() else []
    caso = json.loads(Path(a.caso).read_text(encoding="utf-8")) if a.caso else {}
    idx = [r for r in idx if r["id"] != a.id] + [{"id": a.id, "nome": a.nome or caso.get("nome") or g.get("regiao", a.id),
                                                  "contexto_pt": caso.get("contexto"), "contexto_en": caso.get("contexto_en"),
                                                  "area_ponto": caso.get("ponto"), "bbox": g.get("bbox"),
                                                  "periodo": g.get("periodo"), "malha": f"{a.id}/malha.geojson" if g else None, "pontos": pts,
                                                  "area": area}]
    idx_f.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ok: {a.id}, {len(pts)} ponto(s) -> {APP}")


if __name__ == "__main__":
    main()
