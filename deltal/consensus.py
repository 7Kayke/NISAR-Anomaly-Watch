# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Consenso multissensor por pixel.

Cada evidência é uma máscara booleana (detectou) mais uma máscara de disponibilidade (podia ver:
sem nuvem, fora de sombra/layover, com dado). Confiança = votos / evidências disponíveis.
"""
import numpy as np

from .change import min_area_filter


def combine(evidences, min_votes=2, pixel_area_m2=400.0, min_area_m2=40000.0):
    """evidences: {nome: (detectou, disponivel)}. Devolve votos, disponiveis, confianca, consenso."""
    shape = next(iter(evidences.values()))[0].shape
    votes = np.zeros(shape, "int16")
    avail = np.zeros(shape, "int16")
    for det, av in evidences.values():
        av = np.asarray(av, bool)
        votes += (np.asarray(det, bool) & av).astype("int16")
        avail += av.astype("int16")
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(avail > 0, votes / np.maximum(avail, 1), np.nan)
    cons = votes >= min_votes
    cons, sizes = min_area_filter(cons, max(1, int(round(min_area_m2 / pixel_area_m2))))
    return {"votos": votes, "disponiveis": avail, "confianca": conf, "consenso": cons,
            "consenso_grupos": sizes, "fontes": list(evidences)}
