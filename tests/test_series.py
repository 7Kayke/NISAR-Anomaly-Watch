# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Testes sintéticos da série temporal: degrau, ritmo, alagamento, água e normalização da cena."""
import numpy as np

from deltal import series as S


def _scene(T=20, H=40, W=40, seed=0):
    rng = np.random.default_rng(seed)
    hv = -14 + rng.normal(0, 0.4, (T, H, W))
    hh = -8 + rng.normal(0, 0.4, (T, H, W))
    return hh, hv


def test_classes():
    T = 20
    hh, hv = _scene(T)
    hv += np.linspace(-2, 2, T)[:, None, None]                  # estação: a cena inteira oscila
    hv[12:, 0:10, 0:10] -= 5                                    # queda abrupta na passagem 12 (incêndio)
    hv[:, 10:20, 0:10] += 3 * np.sin(np.arange(T) / T * 4 * np.pi)[:, None, None]   # ritmo agrícola
    hh[[5, 6], 20:30, 0:10] = -24                               # alagamento passageiro
    hh[:, 30:40, 0:10] = -25; hv[:, 30:40, 0:10] = -28          # água permanente
    land = np.ones(hv.shape[1:], bool)
    r = S.classify(hh, hv, land)
    c = r["classe"]
    assert (c[1:9, 1:9] == 3).all()
    assert (r["quebra_indice"][1:9, 1:9] == 12).all()
    assert (c[11:19, 1:9] == 2).mean() > 0.9
    assert (c[21:29, 1:9] == 5).all()
    assert (c[31:39, 1:9] == 6).all()
    assert (c[:, 15:] == 1).mean() > 0.97                      # a estação sozinha não vira mudança


def test_summary_and_truth():
    T = 10
    hh, hv = _scene(T, seed=1)
    hv[6:, 5:15, 5:15] -= 6
    dates = [f"2026-{m:02d}-01" for m in range(1, T + 1)]
    land = np.ones(hv.shape[1:], bool)
    r = S.classify(hh, hv, land)
    truth = np.zeros(hv.shape[1:], bool); truth[5:15, 5:15] = True
    out = S.summarize(r, dates, 400.0, "2026-06-15", truth)
    assert out["indice_primeira_data_pos_evento"] == 6
    assert out["na_cicatriz_quebra_no_intervalo_certo_pct"] > 95
    assert out["na_cicatriz_data_da_quebra_mais_comum"] == "2026-06-01→2026-07-01"


def test_temporal_rgb_shape():
    hh, hv = _scene(9)
    rgb, labels = S.temporal_rgb(hv, [f"d{i}" for i in range(9)])
    assert rgb.shape == (40, 40, 3) and len(labels) == 3


def test_trend_and_moisture():
    T = 20
    rng = np.random.default_rng(3)
    t_days = np.arange(T) * 12
    hh, hv = _scene(T, seed=3)
    hv[:, 0:10, :] += (t_days / 30.44 * -0.5)[:, None, None]    # perde 0,5 dB por mês (degradação lenta)
    wet = rng.uniform(0, 40, T)
    hh[:, 20:30, :] += (wet * 0.1)[:, None, None]                # solo que responde à chuva: +1 dB a cada 10 mm
    land = np.ones(hv.shape[1:], bool)
    r = S.classify(hh, hv, land, t_days=list(t_days))
    assert (r["classe"][1:9, 5:35] == 7).mean() > 0.9
    assert abs(np.median(r["tendencia_db_mes"][1:9]) + 0.5) < 0.1
    rr, sl = S.moisture_response(hh, wet)
    assert np.median(rr[20:30]) > 0.8 and abs(np.median(sl[20:30]) - 0.1) < 0.02
    assert abs(np.median(rr[32:40])) < 0.3
