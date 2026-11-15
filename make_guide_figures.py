# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Figures for the globe guides, in English and Portuguese, sized for the side panel (and readable in the zoom view).

Re-renders from the data whenever the data is in the repository; for the few figures whose raw arrays are not kept
(GUNW pair maps, the MATOPIBA class map, the Norte Fire maps), it crops out the Portuguese text and draws a new title
or legend in each language. Output: docs/globo/pg/fig/<name>_<en|pt>.png

    python make_guide_figures.py
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

R = Path("docs/results")
OUT = Path("docs/globo/pg/fig")
BG, PANEL, INK, INK2, GRID = "#050E26", "#0A1A40", "#E8F0FF", "#A9BCE0", "#1E3366"
TRACK = {"031": "#EAFE07", "132": "#7FE8FF", "010": "#FF8A3D", "111": "#8A94A8", "045": "#EAFE07", "053": "#FF8A3D"}
CLS = {1: ("#8A93A6", "stable", "estável"), 2: ("#2E96F5", "strong seasonal rhythm", "ritmo forte (sazonal)"), 3: ("#FF6B00", "abrupt drop", "queda abrupta"),
       4: ("#EAFE07", "abrupt rise", "subida abrupta"), 5: ("#00C2D1", "short-lived flooding", "alagamento passageiro"), 6: ("#0042A6", "permanent water", "água permanente"),
       7: ("#B5179E", "gradual loss", "queda gradual"), 8: ("#3BB273", "gradual gain", "subida gradual")}
MON = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
       "pt": ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]}


def style():
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "font.family": "DejaVu Sans"})


def dfmt(ax, lang):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: MON[lang][mdates.num2date(x).month - 1]))


def save(fig, name, lang):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}_{lang}.png", dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def d(s):
    s = str(s)
    return datetime.fromisoformat(s[:10]) if "-" in s[:10] else datetime.strptime(s[:8], "%Y%m%d")


# ---------------------------------------------------------------- maps from mapas.tif (Paiaguás)
def class_maps(folder, name, lang, dates):
    import rasterio
    with rasterio.open(folder / "dados" / "mapas.tif") as r:
        cls = r.read(1); brk = r.read(2); px_ha = abs(r.transform.a * r.transform.e) / 1e4
    rgb = np.zeros(cls.shape + (3,))
    for c, (col, *_x) in CLS.items():
        rgb[cls == c] = matplotlib.colors.to_rgb(col)
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.imshow(rgb, interpolation="nearest"); ax.set_axis_off()
    ax.set_title("What changed, pixel by pixel" if lang == "en" else "O que mudou, pixel a pixel")
    hand = [Patch(color=col, label=f"{(en if lang == 'en' else pt)} · {int((cls == c).sum() * px_ha):,} ha".replace(",", "." if lang == "pt" else ","))
            for c, (col, en, pt) in CLS.items() if (cls == c).sum()]
    ax.legend(handles=hand, loc="upper center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    save(fig, f"{name}_classes", lang)
    # when: date of the abrupt step, as intervals
    k = np.where(np.isfinite(brk) & (brk >= 1) & np.isin(cls, [3, 4, 5]), brk, np.nan).astype(float)
    n = len(dates) - 1
    cmap = matplotlib.colors.ListedColormap(["#7FE8FF", "#3FA9F5", "#7CFC8A", "#EAFE07", "#FFB000", "#FF6B00", "#FF3B6B", "#E36BFF", "#FFFFFF", "#B5179E", "#2EC4B6", "#9AA0A6"][:n])
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.imshow(np.ones(cls.shape + (3,)) * matplotlib.colors.to_rgb("#121E3F"))
    im = ax.imshow(k - 1, cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation="nearest"); ax.set_axis_off()
    ax.set_title("When the abrupt change happened" if lang == "en" else "Quando a mudança abrupta aconteceu")
    lab = [f"{d(dates[i]).day} {MON[lang][d(dates[i]).month - 1]} → {d(dates[i + 1]).day} {MON[lang][d(dates[i + 1]).month - 1]}" for i in range(n)]
    hand = [Patch(color=cmap(i), label=f"{lab[i]} · {int(np.nansum(k == i + 1) * px_ha):,} ha".replace(",", "." if lang == "pt" else ",")) for i in range(n) if np.nansum(k == i + 1)]
    ax.legend(handles=hand, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    save(fig, f"{name}_when", lang)


def rgb_temporal(app_dir, name, lang):
    meta = json.loads((app_dir / "app.json").read_text())
    T, H, W = meta["forma"]
    a = np.asarray(Image.open(app_dir / "hv_u8.png").convert("L")).reshape(T, H, W).astype(float)
    a[a == 0] = np.nan; db = -30 + (a - 1) / 254 * 30
    ok = [i for i in range(T) if np.isfinite(db[i]).mean() > 0.3]
    thirds = np.array_split(ok, 3)
    ch = [np.nanmean(db[list(t)], axis=0) for t in thirds]
    rgb = np.dstack([np.clip((c + 26) / 18, 0, 1) for c in ch])
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.imshow(np.nan_to_num(rgb), interpolation="nearest"); ax.set_axis_off()
    ds = meta["datas"]
    rng = lambda t: f"{d(ds[t[0]]).day} {MON[lang][d(ds[t[0]]).month - 1]}–{d(ds[t[-1]]).day} {MON[lang][d(ds[t[-1]]).month - 1]}"
    ax.set_title("Three moments in one image" if lang == "en" else "Três tempos numa imagem")
    hand = [Patch(color=c, label=f"{n} = HV {rng(t)}") for c, n, t in zip(["#FF3B3B", "#3BFF6B", "#3B8BFF"], ["R" if lang == "en" else "Vermelho", "G" if lang == "en" else "Verde", "B" if lang == "en" else "Azul"], thirds)]
    hand.append(Patch(color="#9AA0A6", label="grey = no change" if lang == "en" else "cinza = sem mudança"))
    ax.legend(handles=hand, loc="upper center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.01), labelcolor=INK)
    save(fig, f"{name}_rgb", lang)


# ---------------------------------------------------------------- curves
def rhythm(name, lang, dates, hh, hv, rain, soil, extra_title=""):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    t = [d(x) for x in dates]
    a1.plot(t, hh, "-o", color="#FF8A3D", lw=2, ms=5, label="HH (soil, water)" if lang == "en" else "HH (solo, água)")
    a1.plot(t, hv, "-o", color="#7CFC8A", lw=2, ms=5, label="HV (vegetation)" if lang == "en" else "HV (vegetação)")
    a1.set_ylabel("dB"); a1.grid(alpha=.4); a1.legend(loc="upper left", ncol=2, fontsize=10)
    a1.set_title(("Radar brightness of the area, pass by pass" if lang == "en" else "Brilho do radar na área, passagem a passagem") + extra_title)
    rd = [d(k) for k in sorted(rain)]
    a2.bar(rd, [rain[k] for k in sorted(rain)], width=1, color="#3FA9F5", label="daily rain (mm)" if lang == "en" else "chuva diária (mm)")
    a2.set_ylabel("mm"); a2.grid(alpha=.4)
    b2 = a2.twinx(); sd = [d(k) for k in sorted(soil)]
    b2.plot(sd, [soil[k] for k in sorted(soil)], color="#7CFC8A", lw=1.8, label="soil moisture (m³/m³)" if lang == "en" else "umidade do solo (m³/m³)")
    b2.set_ylim(0, max(0.6, max(soil.values()) * 1.1)); b2.tick_params(colors=INK2); b2.set_ylabel("m³/m³", color=INK2)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); a2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    a2.set_xlim(t[0] - (t[1] - t[0]) / 2, t[-1] + (t[1] - t[0]) / 2); dfmt(a2, lang)
    a2.set_title("Rain and soil moisture (NASA POWER)" if lang == "en" else "Chuva e umidade do solo (NASA POWER)", fontsize=12)
    fig.tight_layout(); save(fig, f"{name}_rhythm", lang)


def point_history(folder, name, lang, rain=None, soil=None):
    rows = [r for r in csv.DictReader(open(folder / "serie_ponto.csv")) if r["passagem_anomala"] != "True"]
    gun = list(csv.DictReader(open(folder / "gunw_ponto.csv")))
    geos = sorted({r["geometria"] for r in rows}, key=lambda g: -sum(1 for r in rows if r["geometria"] == g))
    seen, keep = set(), []
    for g in geos:                                   # neighbouring frames of the same track are the same pass
        k = g[:5]
        if k not in seen:
            seen.add(k); keep.append(g)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 0.8]})
    for g in keep:
        rr = sorted([r for r in rows if r["geometria"] == g], key=lambda r: r["data"]); c = TRACK.get(g[:3], "#FFFFFF")
        tr = f"{'track' if lang == 'en' else 'pista'} {int(g[:3])} {'asc.' if g[4] == 'A' else 'desc.'}"
        x = [d(r["data"]) for r in rr if r["hv_db"]]; y = [float(r["hv_db"]) for r in rr if r["hv_db"]]
        a1.plot(x, y, "-o", color=c, lw=2, ms=5, label=f"HV · {tr}")
    beta = [d(r["data"]) for r in rows if r.get("maturidade") == "beta"]
    if beta:
        a1.axvspan(min(beta), d("2026-01-20"), color="#FFFFFF", alpha=.06); a1.text(min(beta), a1.get_ylim()[1], " BETA", color=INK2, va="top", fontsize=9)
    a1.set_ylabel("HV (dB)"); a1.grid(alpha=.4); a1.legend(fontsize=9.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.6, columnspacing=1)
    a1.set_title("Vegetation (HV) at the point, by pass direction" if lang == "en" else "Vegetação (HV) no ponto, por direção de passagem", pad=26)
    if rain:
        rd = sorted(k for k in rain if k >= min(r["data"] for r in rows)); a2.bar([d(k) for k in rd], [rain[k] for k in rd], width=1, color="#3FA9F5", label="daily rain (mm)" if lang == "en" else "chuva diária (mm)")
    else:
        a2.bar([d(r["data"]) for r in rows], [float(r["chuva_7d_mm"] or 0) for r in rows], width=3, color="#3FA9F5", label="rain, 7 days before (mm)" if lang == "en" else "chuva nos 7 dias antes (mm)")
    a2.set_ylabel("mm"); a2.grid(alpha=.4)
    b2 = a2.twinx()
    if soil:
        sd = sorted(k for k in soil if k >= min(r["data"] for r in rows)); b2.plot([d(k) for k in sd], [soil[k] for k in sd], color="#7CFC8A", lw=1.6, label="soil moisture, NASA POWER" if lang == "en" else "umidade do solo, NASA POWER")
    else:
        sp = sorted(set(r["data"] for r in rows if r["umidade_solo_sup"])); xs, ys = [], []
        for i, k in enumerate(sp):
            if i and (d(k) - d(sp[i - 1])).days > 30:
                xs.append(None); ys.append(np.nan)
            xs.append(d(k)); ys.append(float(next(r["umidade_solo_sup"] for r in rows if r["data"] == k)))
        xs = [x if x else xs[i - 1] for i, x in enumerate(xs)]
        b2.plot(xs, ys, color="#7CFC8A", lw=1.6, label="soil moisture, NASA POWER" if lang == "en" else "umidade do solo, NASA POWER")
    sm = [(d(r["data"]), float(r["nisar_umidade"])) for r in rows if r["nisar_umidade"]]
    if sm:
        b2.plot(*zip(*sm), "D", color="#E36BFF", ms=6, label="NISAR SME2")
    b2.set_ylim(0, 0.7); b2.tick_params(colors=INK2); b2.set_ylabel("m³/m³", color=INK2)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); a2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    a2.set_title("Rain and soil moisture" if lang == "en" else "Chuva e umidade do solo", fontsize=12)
    lanes = {g[:5]: i for i, g in enumerate(keep)}
    for gr in gun:
        k = gr["geometria"][:5]
        if k not in lanes or not gr["coerencia_ponto"]:
            continue
        v = float(gr["coerencia_ponto"]); c = TRACK.get(k[:3], "#FFFFFF")
        a3.plot([d(gr["data1"]), d(gr["data2"])], [v, v], color=c, lw=5, solid_capstyle="butt", alpha=.95)
    a3.axhline(0.3, color="#FF6B6B", lw=1, ls=":"); a3.text(a3.get_xlim()[0], 0.31, " 0.3" if lang == "en" else " 0,3", color="#FF6B6B", fontsize=9)
    a3.set_ylim(0, 1); a3.set_ylabel("0–1"); a3.grid(alpha=.4)
    a3.set_title("GUNW coherence of each 12-day pair (below 0.3 = the ground changed)" if lang == "en" else "Coerência GUNW de cada par de 12 dias (abaixo de 0,3 = o chão mudou)", fontsize=12)
    dfmt(a3, lang); fig.tight_layout(); save(fig, f"{name}_point", lang)


def calendar(folder, name, lang):
    rows = [r for r in csv.DictReader(open(folder / "metadados_ponto.csv")) if r.get("contem_ponto", "True") in ("True", "true", "1")]
    prods = ["GCOV", "GUNW", "SME2"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, pr in enumerate(prods):
        rr = [r for r in rows if r["produto"] == pr]
        for r in rr:
            c = "#8A94A8" if r.get("maturidade") == "beta" else ("#7FE8FF" if r.get("direcao", "").upper().startswith("A") else "#FF8A3D")
            ax.plot(d(r["inicio"]), i, "o", color=c, ms=7)
        ax.text(1.005, i, f"{len(rr)}", transform=ax.get_yaxis_transform(), va="center", color=INK, fontsize=11, fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels(prods); ax.set_ylim(-0.6, 2.6); ax.grid(axis="x", alpha=.4); dfmt(ax, lang)
    ax.set_title("Which NISAR products exist over the point, and when" if lang == "en" else "Quais produtos do NISAR existem sobre o ponto, e quando")
    hand = [Patch(color="#7FE8FF", label="ascending" if lang == "en" else "ascendente"), Patch(color="#FF8A3D", label="descending" if lang == "en" else "descendente"),
            Patch(color="#8A94A8", label="BETA (before calibration)" if lang == "en" else "BETA (antes da calibração)")]
    ax.legend(handles=hand, ncol=3, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); save(fig, f"{name}_calendar", lang)


# ---------------------------------------------------------------- images without raw arrays: replace the Portuguese text
def font(sz, bold=True):
    for f in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def retitle(src, name, titles, crop_top, legend=None, crop_box=None):
    """Cut the original title strip (crop_top px), add a new one per language; optionally paint a legend box over the old one."""
    im = Image.open(src).convert("RGB")
    if crop_box:
        im = im.crop(crop_box)
    im = im.crop((0, crop_top, im.width, im.height))
    for lang, title in titles.items():
        lines = title if isinstance(title, list) else [title]
        f = font(max(15, min(26, im.width // 40))); fs2 = font(max(13, min(22, im.width // 70)))
        lh = f.size + 10
        out = Image.new("RGB", (im.width, im.height + lh * len(lines) + 8), "white")
        out.paste(im, (0, lh * len(lines) + 8)); dr = ImageDraw.Draw(out)
        if isinstance(title, list) and len(title) > 1 and crop_box is not None:
            pass
        for i, ln in enumerate(lines):
            if isinstance(ln, tuple):                       # (x, text) for multi-panel headers
                for x, tx in ln:
                    dr.text((x, 8 + i * lh), tx, fill="#0B1530", font=fs2)
            else:
                dr.text((10, 6 + i * lh), ln, fill="#0B1530", font=f)
        if legend:
            (x0, y0, x1, y1), items = legend[0], legend[1][lang]
            y0 += lh * len(lines) + 8 - crop_top; y1 += lh * len(lines) + 8 - crop_top
            dr.rectangle((x0, y0, x1, y1), fill="#FFFFFF", outline="#B8C3D6")
            fs = font(13, bold=False); step = (y1 - y0 - 8) / len(items)
            for j, (col, tx) in enumerate(items):
                yy = y0 + 4 + j * step; dr.rectangle((x0 + 8, yy + 3, x0 + 34, yy + step - 4), fill=col); dr.text((x0 + 42, yy + 1), tx, fill="#0B1530", font=fs)
        OUT.mkdir(parents=True, exist_ok=True); out.save(OUT / f"{name}_{lang}.png")


def main():
    style()
    clima = json.loads(Path("docs/globo/pg/clima.json").read_text())
    rain_all = clima["chuva_mm"]; soil_all = clima["umidade_solo"]
    p = R / "paiaguas" / "031_A_169"
    app = json.loads((p / "app" / "app.json").read_text()) if (p / "app" / "app.json").exists() else json.loads(Path("docs/globo/pg/031_A_169/app.json").read_text())
    dates = app["datas"]
    rows = list(csv.DictReader(open(p / "dados" / "serie_area.csv")))
    for lang in ("en", "pt"):
        class_maps(p, "paiaguas", lang, dates)
        rgb_temporal(Path("docs/globo/pg/031_A_169"), "paiaguas", lang)
        rr = {k: v for k, v in rain_all.items() if dates[0] <= k <= dates[-1]}; ss = {k: v for k, v in soil_all.items() if dates[0] <= k <= dates[-1]}
        rhythm("paiaguas", lang, [r["data"] for r in rows], [float(r["hh_mediana_db"]) for r in rows], [float(r["hv_mediana_db"]) for r in rows], rr, ss)
        point_history(R / "ponto_paiaguas", "paiaguas", lang, rain_all, soil_all)
        point_history(R / "ponto_tocantins", "tocantins", lang)
        calendar(R / "ponto_tocantins", "tocantins", lang)
        sj = json.loads((R / "cerrado_mat" / "serie.json").read_text())
        cl = sj["clima"]; rain = {k: v for k, v in cl["PRECTOTCORR"].items()} if isinstance(cl["PRECTOTCORR"], dict) else {}
        soil = {k: v for k, v in cl["GWETTOP"].items()} if isinstance(cl["GWETTOP"], dict) else {}
        rain = {k[:4] + "-" + k[4:6] + "-" + k[6:] if "-" not in k else k: v for k, v in rain.items()}
        soil = {k[:4] + "-" + k[4:6] + "-" + k[6:] if "-" not in k else k: v for k, v in soil.items()}
        sd = sj["datas"]
        rain = {k: v for k, v in rain.items() if sd[0] <= k <= sd[-1] and v >= 0}; soil = {k: v for k, v in soil.items() if sd[0] <= k <= sd[-1] and v >= 0}
        rhythm("matopiba", lang, sd, sj["curva_cena_hh"], sj["curva_cena_hv"], rain, soil)
    # GUNW pairs: two panels, Portuguese titles in the top ~32 px
    for f, nm, ti in [("par_20260908_20260920_031A.png", "pair_sep", {"en": "Sep 8 → Sep 20, 2026 · track 31 ascending", "pt": "08/09 → 20/09/2026 · pista 31 ascendente"}),
                      ("par_20260628_20260710_031A.png", "pair_jul", {"en": "Jun 28 → Jul 10, 2026 · track 31 ascending", "pt": "28/06 → 10/07/2026 · pista 31 ascendente"})]:
        src = R / "paiaguas" / "gunw" / f; W = Image.open(src).width
        retitle(src, f"paiaguas_{nm}", {l: [ti[l], ((10, "Displacement (cm, relative)" if l == "en" else "Deslocamento (cm, relativo)"), (W // 2 + 30, "Coherence (0 = changed, 1 = same)" if l == "en" else "Coerência (0 = mudou, 1 = igual)"))] for l in ("en", "pt")}, 30)
    retitle(R / "norte_fire_funil" / "nisar_mudanca_hv.png", "norte_hv", {"en": "NISAR HV change (dB) · red = darkened", "pt": "Mudança no HV do NISAR (dB) · vermelho = escureceu"}, 34)
    retitle(R / "norte_fire_funil" / "consenso.png", "norte_consensus", {"en": "Orange: radar + optical agree · white: Sentinel-2 scar", "pt": "Laranja: radar + óptico concordam · branco: cicatriz Sentinel-2"}, 34)
    prev = Path("docs/globo/pg/norte/previa_nisar.png"); pw = Image.open(prev).width
    retitle(prev, "norte_preview", {"en": [((20, "Oct 14 · 2 days before"), (584, "Oct 26 · 10 days after"), (1148, "Change in green (≈ HV)"))],
                                     "pt": [((20, "14/10 · 2 dias antes"), (584, "26/10 · 10 dias depois"), (1148, "Mudança no verde (≈ HV)"))]}, 0, crop_box=(0, 108, pw, 592))
    retitle(R / "cerrado_mat" / "quando_mudou.png", "matopiba_when", {"en": "When the abrupt change happened", "pt": "Quando a mudança abrupta aconteceu"}, 30)
    lg = {l: [(c, (en if l == "en" else pt)) for c, (c2, en, pt) in [(v[0], v) for v in CLS.values()]] for l in ("en", "pt")}
    retitle(R / "cerrado_mat" / "classes.png", "matopiba_classes", {"en": "Change classes · Jun 27 – Oct 25, 2026 (10 passes)", "pt": "Classes de mudança · 27/06 – 25/10/2026 (10 passagens)"}, 30, legend=((133, 460, 322, 624), lg))
    print("figures:", len(list(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
