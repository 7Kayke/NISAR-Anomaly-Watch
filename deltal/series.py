# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Série temporal do radar ("a valsa"): ritmo de cada pixel, quando ele quebrou e de que tipo foi a mudança.

Entrada: pilhas (T, H, W) de HH e HV em dB, uma camada por passagem do NISAR na MESMA geometria.

1. normalização por data: tira a mediana espacial da terra em cada data. O que sobra são os
   desvios LOCAIS; o ritmo da paisagem inteira (estação, chuva, calibração) vai para a curva da cena.
2. quebra: para cada pixel e cada corte t, diferença das médias depois-antes e o t de Welch
   simplificado (desvio combinado). Fica o corte com maior |t|.
3. ritmo: desvio-padrão dos resíduos depois de tirar a quebra (o que oscila sem ser um degrau).
4. classes (prioridade de cima para baixo):
     6 água permanente      HH mediana < água
     3 queda abrupta        HV caiu >= queda_db com |t| >= t_min   (fogo, corte, colheita definitiva)
     4 subida abrupta       HV subiu >= queda_db com |t| >= t_min  (rebrota forte, obra, água sob vegetação)
     5 alagamento passageiro HH < água em alguma data, mas a mediana não é água
     2 ritmo forte          resíduo >= ritmo_db                     (agricultura, áreas úmidas, sazonal)
     1 estável
     0 sem dado
"""
from datetime import datetime

import numpy as np

CLASSES = {0: "sem dado", 1: "estável", 2: "ritmo forte (sazonal)", 3: "queda abrupta",
           4: "subida abrupta", 5: "alagamento passageiro", 6: "água permanente",
           7: "queda gradual", 8: "subida gradual"}
CLASS_COLORS = {0: "#00000000", 1: "#8A93A6", 2: "#2E96F5", 3: "#FF6B00", 4: "#EAFE07",
                5: "#00C2D1", 6: "#0042A6", 7: "#B5179E", 8: "#3BB273"}
PARAMS = {"queda_db": 2.5, "t_min": 4.0, "ritmo_db": 1.5, "agua_db": -18.0, "min_lados": 2,
          "min_validas": 5, "gradual_db": 2.0, "t_tendencia": 4.0}


def normalize(stack, land):
    """Subtrai a mediana espacial (só terra) de cada data. Devolve (normalizada, curva_da_cena)."""
    curve = np.array([np.nanmedian(s[land]) if np.any(land & np.isfinite(s)) else np.nan for s in stack])
    return stack - curve[:, None, None], curve


def breakpoint(stack, min_side=2):
    """Melhor degrau no tempo por pixel. Devolve (indice_do_primeiro_depois, delta_db, t, residuo_db)."""
    T = stack.shape[0]
    x = stack.reshape(T, -1)
    ok = np.isfinite(x)
    xz = np.where(ok, x, 0.0)
    cs, cn, cq = np.cumsum(xz, 0), np.cumsum(ok, 0), np.cumsum(xz * xz, 0)
    S, N, Q = cs[-1], cn[-1], cq[-1]
    best_t = np.zeros(x.shape[1]); best_k = np.zeros(x.shape[1], int); best_d = np.zeros(x.shape[1])
    best_r = np.full(x.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, T):                      # depois = [k, T)
            n1, s1, q1 = cn[k - 1], cs[k - 1], cq[k - 1]
            n2, s2, q2 = N - n1, S - s1, Q - q1
            m1, m2 = s1 / n1, s2 / n2
            sse = (q1 - n1 * m1 ** 2) + (q2 - n2 * m2 ** 2)
            sd = np.sqrt(sse / np.maximum(n1 + n2 - 2, 1))
            sd = np.maximum(sd, 0.3)               # piso: speckle residual depois do filtro
            t = (m2 - m1) / (sd * np.sqrt(1 / n1 + 1 / n2))
            good = (n1 >= min_side) & (n2 >= min_side) & np.isfinite(t)
            upd = good & (np.abs(t) > np.abs(best_t))
            best_t[upd], best_k[upd], best_d[upd], best_r[upd] = t[upd], k, (m2 - m1)[upd], sd[upd]
    shp = stack.shape[1:]
    return best_k.reshape(shp), best_d.reshape(shp), best_t.reshape(shp), best_r.reshape(shp)


def trend(stack, t_days):
    """Tendência linear por pixel (mínimos quadrados, ignora NaN). Devolve (dB por mês, t da inclinação)."""
    T = stack.shape[0]
    x = stack.reshape(T, -1)
    ok = np.isfinite(x)
    t = (np.asarray(t_days, float) / 30.44)[:, None] * ok
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = t.sum(0) / n
        xm = np.where(ok, x, 0).sum(0) / n
        dt = np.where(ok, t - tm, 0)
        dx = np.where(ok, x - xm, 0)
        sxx = (dt * dt).sum(0)
        b = (dt * dx).sum(0) / sxx
        res = np.where(ok, dx - b * dt, 0)
        se = np.sqrt((res * res).sum(0) / np.maximum(n - 2, 1) / sxx)
        tt = b / np.maximum(se, 1e-6)
    shp = stack.shape[1:]
    return b.reshape(shp), tt.reshape(shp)


def moisture_response(stack, wet):
    """Correlação (r) e inclinação (dB por unidade do índice) entre o radar e um índice de umidade por data.
    É a valsa invisível da água: solo e vegetação mais úmidos devolvem mais energia (constante dielétrica)."""
    T = stack.shape[0]
    w = np.asarray(wet, float)
    x = stack.reshape(T, -1)
    ok = np.isfinite(x) & np.isfinite(w)[:, None]
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wm = np.where(ok, w[:, None], 0).sum(0) / n
        xm = np.where(ok, x, 0).sum(0) / n
        dw = np.where(ok, w[:, None] - wm, 0)
        dx = np.where(ok, x - xm, 0)
        cov = (dw * dx).sum(0)
        r = cov / np.sqrt((dw * dw).sum(0) * (dx * dx).sum(0))
        slope = cov / (dw * dw).sum(0)
    r[n < 4] = np.nan
    shp = stack.shape[1:]
    return r.reshape(shp), slope.reshape(shp)


def classify(hh_db, hv_db, land, params=None, valid=None, t_days=None):
    """land: pixels usados na normalização (terra). valid: pixels que recebem classe (padrão: todos com dado)."""
    p = dict(PARAMS, **(params or {}))
    hv_n, hv_curve = normalize(hv_db, land)
    hh_n, hh_curve = normalize(hh_db, land)
    nvalid = np.isfinite(hv_db).sum(0)
    k, d, t, resid = breakpoint(hv_n, p["min_lados"])
    hh_med = np.nanmedian(hh_db, 0)
    hh_min = np.nanmin(np.where(np.isfinite(hh_db), hh_db, np.inf), 0)
    cls = np.ones(hv_db.shape[1:], "uint8")
    cls[resid >= p["ritmo_db"]] = 2
    cls[(hh_min < p["agua_db"]) & ~(hh_med < p["agua_db"])] = 5
    slope = tt = None
    if t_days is not None:
        slope, tt = trend(hv_n, t_days)
        span = (max(t_days) - min(t_days)) / 30.44
        grad = np.abs(tt) >= p["t_tendencia"]
        cls[grad & (slope * span <= -p["gradual_db"])] = 7
        cls[grad & (slope * span >= p["gradual_db"])] = 8
    strong = np.abs(t) >= p["t_min"]
    cls[strong & (d <= -p["queda_db"])] = 3
    cls[strong & (d >= p["queda_db"])] = 4
    cls[hh_med < p["agua_db"]] = 6
    if valid is None:
        valid = np.ones(cls.shape, bool)
    cls[(nvalid < p["min_validas"]) | ~valid] = 0
    when = np.where(np.isin(cls, (3, 4)), k, -1)
    return {"classe": cls, "quebra_indice": when, "delta_db": d, "t": t, "residuo_db": resid,
            "tendencia_db_mes": slope, "tendencia_t": tt,
            "curva_cena_hv": hv_curve, "curva_cena_hh": hh_curve, "hv_norm": hv_n, "params": p}


def temporal_rgb(hv_db, dates, split=None):
    """Composição de 3 tempos: R = começo, G = meio, B = fim (médias em dB).
    Cinza = não mudou. Vermelho = era forte no começo e caiu. Azul = ficou forte no fim."""
    T = len(dates)
    a, b = (T // 3, 2 * T // 3) if split is None else split
    parts = [hv_db[:max(a, 1)], hv_db[max(a, 1):max(b, a + 1)], hv_db[max(b, a + 1):]]
    means = [np.nanmean(p, 0) for p in parts]
    v = np.concatenate([m[np.isfinite(m)] for m in means])
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
    rgb = np.dstack([np.clip((m - lo) / (hi - lo + 1e-9), 0, 1) for m in means])
    labels = [f"{dates[0]}..{dates[max(a, 1) - 1]}", f"{dates[max(a, 1)]}..{dates[max(b, a + 1) - 1]}",
              f"{dates[max(b, a + 1)]}..{dates[-1]}"]
    return np.nan_to_num(rgb), labels


def summarize(res, dates, px_area_m2, event_date=None, truth=None):
    cls = res["classe"]
    out = {"classes_ha": {CLASSES[c]: round(float((cls == c).sum() * px_area_m2 / 1e4), 2) for c in CLASSES if c},
           "datas": dates}
    k = res["quebra_indice"]
    hist = {}
    for i in range(1, len(dates)):
        n = int(((k == i) & (cls == 3)).sum())
        if n:
            hist[f"{dates[i - 1]}→{dates[i]}"] = round(n * px_area_m2 / 1e4, 2)
    out["quedas_por_intervalo_ha"] = hist
    if event_date:
        ev = datetime.fromisoformat(event_date)
        idx = next((i for i, d in enumerate(dates) if datetime.fromisoformat(d) > ev), None)
        out["indice_primeira_data_pos_evento"] = idx
        if idx is not None:
            hit = (cls == 3) & (k == idx)
            out["queda_no_intervalo_do_evento_ha"] = round(float(hit.sum() * px_area_m2 / 1e4), 2)
            res["queda_evento"] = hit
            if truth is not None and truth.any():
                kt = k[truth & (cls == 3)]
                out["na_cicatriz_classe_queda_pct"] = round(100.0 * float((cls[truth] == 3).mean()), 1)
                if kt.size:
                    vals, cnt = np.unique(kt, return_counts=True)
                    out["na_cicatriz_data_da_quebra_mais_comum"] = f"{dates[vals[cnt.argmax()] - 1]}→{dates[vals[cnt.argmax()]]}"
                    out["na_cicatriz_quebra_no_intervalo_certo_pct"] = round(100.0 * float((kt == idx).mean()), 1)
    return out
