"""API REST do Delta L (FastAPI). Rodar no computador A:  uvicorn api.app:app --port 8000

GET  /v1/collections                     fontes do funil e o que cada uma mede
GET  /v1/search?bbox=w,s,e,n&date=AAAA-MM-DD   roda SÓ o funil (catálogo -> pareados), sem baixar pixels
POST /v1/analyses {"case": {...}}        dispara a análise completa (run_funnel) em segundo plano
GET  /v1/analyses                        lista análises
GET  /v1/analyses/{id}                   status + ficha técnica
GET  /v1/analyses/{id}/files/{nome}      figuras e ficha (PNG, JSON, CSV)
A mesma forma de resposta é publicada como JSON estático em docs/api/v1/ (GitHub Pages) para o usuário final.
Parte deste código teve apoio de IA (Claude).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
CASES = ROOT / "cases"
COLLECTIONS = json.loads((ROOT / "docs" / "api" / "v1" / "collections.json").read_text(encoding="utf-8"))

app = FastAPI(title="Delta L API", version="1")


class Case(BaseModel):
    nome: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    evento: dict = Field(description='{"data": "AAAA-MM-DD", "descricao": "...", "area_oficial_ha": opcional}')
    resolucao_m: int = 20


@app.get("/v1/collections")
def collections():
    return COLLECTIONS


@app.get("/v1/search")
def search(bbox: str, date: str, nisar: bool = False):
    from deltal import collect
    from deltal.grid import make_grid
    w, s, e, n = [float(v) for v in bbox.split(",")]
    if not (w < e and s < n) or (e - w) * (n - s) > 1.0:
        raise HTTPException(400, "bbox inválida ou maior que 1 grau²")
    grid = make_grid((w, s, e, n), 20)
    out = []
    for name, fn in [("sentinel2", collect.sentinel2), ("landsat", collect.landsat), ("sentinel1", collect.sentinel1),
                     ("aster", collect.aster)]:
        try:
            fun, best = fn(grid, date)
            out.append(fun.summary() | {"par": _pair(best)})
        except Exception as ex:
            out.append({"fonte": name, "erro": str(ex)})
    if nisar:
        try:
            from deltal import sources_nisar as sn
            sn.login()
            fun, best = sn.funnel(grid, date)
            out.append(fun.summary() | {"par": None if not best else
                                        [b[1]["umm"]["GranuleUR"] for b in best[1:3]] + [best[3]]})
        except Exception as ex:
            out.append({"fonte": "nisar", "erro": str(ex)})
    return {"bbox": [w, s, e, n], "data_evento": date, "funil": out}


def _pair(best):
    if not best:
        return None
    return {"antes": best[1].ids, "depois": best[2].ids, "intervalo_dias": best[3]}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def _run(case_path, out):
    subprocess.run([sys.executable, str(ROOT / "run_funnel.py"), "--case", str(case_path), "--out", str(out)],
                   cwd=ROOT, check=False)


@app.post("/v1/analyses", status_code=202)
def create(case: Case, bg: BackgroundTasks):
    aid = _slug(case.nome)
    path = CASES / f"{aid}.json"
    path.write_text(case.model_dump_json(indent=2), encoding="utf-8")
    bg.add_task(_run, path, RESULTS / aid)
    return {"id": aid, "status": "na_fila", "url": f"/v1/analyses/{aid}"}


@app.get("/v1/analyses")
def list_analyses():
    if not RESULTS.is_dir():
        return []
    return [{"id": p.name, "url": f"/v1/analyses/{p.name}"} for p in sorted(RESULTS.iterdir()) if p.is_dir()]


@app.get("/v1/analyses/{aid}")
def get_analysis(aid: str):
    d = RESULTS / _slug(aid)
    if not d.is_dir():
        raise HTTPException(404, "análise não encontrada")
    f = d / "ficha_tecnica.json"
    if not f.exists():
        return {"id": aid, "status": "rodando"}
    return {"id": aid, "status": "concluida", "ficha": json.loads(f.read_text(encoding="utf-8")),
            "arquivos": [f"/v1/analyses/{aid}/files/{p.name}" for p in sorted(d.iterdir()) if p.is_file()]}


@app.get("/v1/analyses/{aid}/files/{name}")
def get_file(aid: str, name: str):
    p = (RESULTS / _slug(aid) / name).resolve()
    if RESULTS.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)
