"""Saídas: figura de antes/depois/mudança, ficha técnica (JSON e CSV) e texto do resultado."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Paleta do guia de marca do Space Apps
DEEP, ELECTRIC, YONDER, YELLOW = "#07173F", "#0042A6", "#2E96F5", "#EAFE07"


def _stretch(a, lo=2, hi=98):
    a = np.asarray(a, dtype="float64")
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-12), 0, 1)


def false_color(refl):
    rgb = np.dstack([_stretch(refl["B12"]), _stretch(refl["B08"]), _stretch(refl["B04"])])
    return np.nan_to_num(rgb)


def save_png(path, arr, cmap="RdBu", vmin=-6, vmax=6, title=None):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=DEEP, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlay_png(path, base_rgb, truth=None, layers=None, title=None):
    """Base em cor (falsa cor S2 ou HV), contorno da verdade de campo e camadas detectadas."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.imshow(base_rgb)
    colors = {"vegetacao": "#FF6B00", "agua": YONDER, "dano": YELLOW}
    for name, m in (layers or {}).items():
        if m is not None and np.any(m):
            ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([colors.get(name, "#FF00FF")]),
                      alpha=0.55, interpolation="nearest")
    if truth is not None and np.any(truth):
        ax.contour(truth.astype(float), levels=[0.5], colors=["white"], linewidths=1.4)
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=DEEP, fontsize=11)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_ficha(outdir, ficha):
    outdir = Path(outdir)
    ficha = _clean(ficha)
    (outdir / "ficha_tecnica.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            for i, v in enumerate(o):
                walk(f"{prefix}[{i}]", v)
        else:
            rows.append((prefix, json.dumps(o, ensure_ascii=False) if isinstance(o, list) else o))
    walk("", ficha)
    with open(outdir / "ficha_tecnica.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["campo", "valor"])
        w.writerows(rows)
    return ficha


def _metric_line(m):
    return (f"precisão {m.get('precisao')}, sensibilidade {m.get('sensibilidade')}, F1 {m.get('f1')}, "
            f"IoU {m.get('iou')}, falso alarme {m.get('falso_alarme')}")


def narrative(ficha):
    """Texto em linguagem simples a partir dos números (sem inventar nada)."""
    lines = [f"# {ficha.get('nome', 'Análise')}", ""]
    ev = ficha.get("evento", {})
    if ev:
        lines.append(f"Evento de referência: {ev.get('descricao')} ({ev.get('data')}).")
    if ficha.get("funil"):
        lines += ["", "## Funil de coleta", "", "| Fonte | " + " | ".join(["catálogo", "cobre a área", "qualidade", "pareados"]) + " | aproveitamento |",
                  "|---|---|---|---|---|---|"]
        for f in ficha["funil"]:
            n = {e["etapa"]: e["n"] for e in f["etapas"]}
            cells = [str(n.get(k, "-")) for k in ["catalogo", "cobre_area", "qualidade", "pareados"]]
            tot = f.get("aproveitamento_total_pct")
            lines.append(f"| {f['fonte']} | " + " | ".join(cells) + f" | {'-' if tot is None else str(tot) + '%'} |")
    for src, res in ficha.get("resultados", {}).items():
        lines += ["", f"## {src}"]
        st = res.get("stats") or {}
        if st:
            lines.append(f"- Perda de vegetação detectada: {st.get('vegetacao_area_ha')} ha; água nova: {st.get('agua_area_ha')} ha.")
        for k, v in res.items():
            if k.startswith("metricas") and isinstance(v, dict):
                lines.append(f"- {k.replace('_', ' ')} contra a verdade de campo: {_metric_line(v)}.")
            elif k not in ("stats",) and not isinstance(v, (dict, list)):
                lines.append(f"- {k.replace('_', ' ')}: {v}")
    ctx = ficha.get("contexto", {}).get("chuva")
    if ctx:
        lines += ["", f"Chuva entre as datas (NASA POWER): {ctx.get('total_mm')} mm."]
    if ficha.get("erros"):
        lines += ["", "## Etapas que falharam", ""] + [f"- {k}: {v}" for k, v in ficha["erros"].items()]
    lines += ["", "Parte do código e do texto teve apoio de IA (Claude); números calculados pelo pipeline e revisados pela equipe."]
    return "\n".join(lines)
