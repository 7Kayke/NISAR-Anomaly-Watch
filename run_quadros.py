# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Malha REAL de quadros do NISAR para o globo: lê só os METADADOS do catálogo da NASA (CMR), sem baixar produtos.

Uso:
    python run_quadros.py --out docs/globo/quadros.json                 # planeta inteiro
    python run_quadros.py --bbox -82 -56 -34 13 --out docs/globo/quadros.json   # só a América do Sul

Para cada quadro (pista, direção, número do quadro) guarda:
  contorno (polígono do primeiro arquivo, simplificado), primeiro e último arquivo, data de cada um,
  quantas passagens GCOV existem e a prévia (browse) do primeiro e do último, se a NASA publicar.
Não precisa de login: a busca no CMR é pública.
"""
import argparse
import json
import time
from pathlib import Path

import requests

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
COLS = ["NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_BETA_V1"]


def att(u, name):
    for a in u.get("AdditionalAttributes", []):
        if a.get("Name") == name and a.get("Values"):
            return a["Values"][0]
    return None


def browse(u):
    for r in u.get("RelatedUrls", []):
        url = r.get("URL", "")
        if r.get("Type") == "GET RELATED VISUALIZATION" and url.lower().endswith((".png", ".jpg", ".jpeg")):
            return url
    return None


def polygon(u):
    try:
        pts = u["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        ring = [[round(p["Longitude"], 3), round(p["Latitude"], 3)] for p in pts]
        if len(ring) > 12:                       # simplifica: o globo não precisa de mais que isso
            step = max(1, len(ring) // 10)
            ring = ring[::step] + [ring[0]]
        return ring
    except (KeyError, IndexError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/globo/quadros.json")
    ap.add_argument("--bbox", nargs=4, type=float, help="oeste sul leste norte (opcional)")
    a = ap.parse_args()
    frames, total, s = {}, 0, requests.Session()
    for col in COLS:
        params = {"provider": "ASF", "short_name": col, "page_size": 2000, "sort_key": "start_date"}
        if a.bbox:
            params["bounding_box"] = ",".join(str(v) for v in a.bbox)
        after, page = None, 0
        while True:
            h = {"CMR-Search-After": after} if after else {}
            for tent in range(4):
                try:
                    r = s.get(CMR, params=params, headers=h, timeout=120)
                    r.raise_for_status()
                    break
                except Exception as e:
                    print(f"  tentativa {tent + 1} falhou: {e}", flush=True); time.sleep(5)
            else:
                raise SystemExit("CMR não respondeu")
            js = r.json(); items = js.get("items", [])
            page += 1
            print(f"{col}: página {page}, {len(items)} arquivos (total na coleção {js.get('hits')})", flush=True)
            for it in items:
                u = it["umm"]
                tr, dr, fr = att(u, "TRACK_NUMBER"), att(u, "ASCENDING_DESCENDING"), att(u, "FRAME_NUMBER")
                if tr is None or fr is None:
                    continue
                d = "A" if str(dr).upper().startswith("A") else "D"
                key = f"{int(tr):03d}_{d}_{int(fr):03d}"
                day = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
                f = frames.get(key)
                if f is None:
                    ring = polygon(u)
                    if not ring:
                        continue
                    f = frames[key] = {"id": key, "t": int(tr), "d": d, "f": int(fr), "p": ring, "n": 0,
                                       "p0": day, "g0": u["GranuleUR"], "b0": browse(u), "p1": day, "g1": u["GranuleUR"], "b1": browse(u), "dias": []}
                if day < f["p0"]:
                    f.update(p0=day, g0=u["GranuleUR"], b0=browse(u))
                if day >= f["p1"]:
                    f.update(p1=day, g1=u["GranuleUR"], b1=browse(u))
                if day not in f["dias"]:
                    f["dias"].append(day); f["n"] += 1
                total += 1
            after = r.headers.get("CMR-Search-After")
            if not items or not after:
                break
    for f in frames.values():
        f.pop("dias")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_compact(frames.values(), total, out)


def write_compact(frames, total, out):
    """Formato enxuto para o navegador (~3x menor): listas em vez de objetos, coordenadas com 2 casas (~1 km)
    e a prévia guardada só como o nome da coleção quando segue o padrão do ASF (a URL é remontada no app)."""
    import re
    pat = re.compile(r"^https://nisar\.asf\.earthdatacloud\.nasa\.gov/BROWSE/([A-Z0-9_]+)/(.+?)/\2_LATLON\.png$")
    cols, rows = [], []
    def enc(url, gran):
        m = pat.match(url or "")
        if m and m.group(2) == gran:
            if m.group(1) not in cols:
                cols.append(m.group(1))
            return cols.index(m.group(1))
        return url or ""
    for f in frames:
        ring = [c for p in f["p"] for c in (round(p[0], 2), round(p[1], 2))]
        rows.append([f["t"], f["d"], f["f"], f["n"], ring, f["p0"], f["p1"], f["g0"], f["g1"], enc(f["b0"], f["g0"]), enc(f["b1"], f["g1"])])
    out.write_text(json.dumps({"fonte": "NASA CMR (ASF), coleções " + ", ".join(COLS), "gerado": time.strftime("%Y-%m-%d %H:%M"), "arquivos": total,
                               "formato": "t,d,f,n,anel,p0,p1,g0,g1,b0,b1", "browse": "https://nisar.asf.earthdatacloud.nasa.gov/BROWSE/{col}/{g}/{g}_LATLON.png",
                               "colecoes": cols, "q": rows}, separators=(",", ":")), encoding="utf-8")
    print(f"pronto: {len(rows)} quadros a partir de {total} arquivos -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
