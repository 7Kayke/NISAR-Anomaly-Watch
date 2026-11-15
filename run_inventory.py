# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Inventário de todos os produtos NISAR de uma área (só metadados, nada é baixado; leva ~1-3 min).

Uso:
    python run_inventory.py --case cases/cerrado_matopiba.json --out docs/results/cerrado_matopiba
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

from deltal import inventory as inv
from deltal import sources_nisar as sn
from run_series import BETA_END, GAPS, fit_to_frame, log


def run(case, out, start, end, bbox=None, geometry=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    bbox = bbox or case["bbox"]
    log("inventário de todos os produtos NISAR ...")
    rows, funnels, cols = inv.inventory(bbox, start, end, geometry=geometry, log=lambda m: log(m))
    keys = sorted({k for r in rows for k in r})
    with open(out / "inventario.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    summary = {"area_bbox": bbox, "quadro": geometry, "periodo": [start, end], "colecoes": cols, "funil_por_produto": funnels,
               "total_arquivos": len(rows),
               "por_produto": {f["fonte"]: f["etapas"][-1]["n"] for f in funnels}}
    (out / "inventario.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    inv.calendar_png(out / "calendario_produtos.png", rows, start, end, GAPS)
    lines = ["| Produto | catálogo | cobre a área | mesmo quadro | uma por passagem |", "|---|---|---|---|---|"]
    for f in funnels:
        n = {e["etapa"]: e["n"] for e in f["etapas"]}
        lines.append(f"| {f['fonte']} | {n.get('catalogo')} | {n.get('cobre_area')} | {n.get('mesmo_quadro')} | {n.get('uma_por_passagem')} |")
    (out / "inventario.md").write_text("\n".join(lines), encoding="utf-8")
    log("inventário pronto")
    print("\n".join(lines))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--start"); ap.add_argument("--end")
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text(encoding="utf-8"))
    cfg = case.get("serie", {})
    start = a.start or cfg.get("inicio", BETA_END)
    end = a.end or cfg.get("fim", date.today().isoformat())
    sn.login()
    bbox, geometry = case["bbox"], None
    if case.get("nisar_quadro"):
        q = case["nisar_quadro"]
        geometry = (int(q["pista"]), q["direcao"], int(q["quadro"]))
        bbox, _ = fit_to_frame(bbox, start, end, geometry, sn)
    run(case, a.out, start, end, bbox, geometry)


if __name__ == "__main__":
    main()
