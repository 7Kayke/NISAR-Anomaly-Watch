# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Funil de coleta unificado.

Etapas, para cada fonte:
  1. catalogo    : itens que o catálogo devolve para a área e o período
  2. cobre_area  : itens cujo polígono cobre pelo menos MIN_COVER da área
  3. qualidade   : itens que passam no filtro de qualidade da fonte
                   (nuvem dentro da área, órbita/geometria definida, banda térmica presente...)
  4. pareados    : itens que formam um par antes/depois comparável (mesma geometria)
Cada etapa guarda a contagem e a % sobre a etapa anterior ("aproveitamento").
"""
from dataclasses import dataclass, field

from shapely.geometry import box, shape

MIN_COVER = 0.9
# O "depois" precisa começar DEPOIS que o evento acabou de acontecer: uma cena do próprio dia
# do início do incêndio mostra a área ainda queimando (Incêndio Norte: S2 de 16/10 viu só 36 ha,
# a de 19/10 viu ~227 ha). Ajustável por caso em "evento.dias_apos_minimo".
POST_MIN_DAYS = 2


@dataclass
class Stage:
    name: str
    items: list = field(default_factory=list)
    note: str = ""


@dataclass
class Funnel:
    source: str
    collection: str
    stages: list = field(default_factory=list)

    def add(self, name, items, note=""):
        self.stages.append(Stage(name, list(items), note))
        return list(items)

    def summary(self):
        out, prev = [], None
        for st in self.stages:
            n = len(st.items)
            pct = None if prev in (None, 0) else round(100.0 * n / prev, 1)
            out.append({"etapa": st.name, "n": n, "aproveitamento_pct": pct, "nota": st.note})
            prev = n
        first = self.stages[0] if self.stages else None
        last = self.stages[-1] if self.stages else None
        total = None
        if first and last and len(first.items):
            total = round(100.0 * len(last.items) / len(first.items), 1)
        return {"fonte": self.source, "colecao": self.collection, "etapas": out, "aproveitamento_total_pct": total}


def coverage(item_geometry, bbox):
    aoi = box(*bbox)
    geom = shape(item_geometry)
    return geom.intersection(aoi).area / aoi.area if aoi.area else 0.0


def covering(items, bbox, min_cover=MIN_COVER, geom_of=lambda it: it.geometry):
    return [it for it in items if coverage(geom_of(it), bbox) >= min_cover]


def pair_by_key(items, event_dt, key, date_of, post_min_days=None):
    """Par antes/depois com o menor intervalo, dentro do mesmo grupo de geometria (key).

    antes: data < dia do evento (00:00). depois: data >= dia do evento + post_min_days.
    date_of(item) e event_dt são datetimes. Devolve (chave, antes, depois, intervalo_dias) ou None.
    """
    from datetime import timedelta
    day0 = event_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    post0 = day0 + timedelta(days=POST_MIN_DAYS if post_min_days is None else post_min_days)
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    best = None
    for k, its in groups.items():
        pre = [i for i in its if date_of(i) < day0]
        post = [i for i in its if date_of(i) >= post0]
        if pre and post:
            a = max(pre, key=date_of)
            b = min(post, key=date_of)
            gap = (date_of(b) - date_of(a)).total_seconds() / 86400.0
            if best is None or gap < best[3]:
                best = (k, a, b, round(gap, 2))
    return best
