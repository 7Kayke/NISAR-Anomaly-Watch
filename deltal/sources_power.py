# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Chuva diária do NASA POWER (contexto: umidade muda o brilho do radar)."""
import requests

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def rain_between(lon, lat, start_yyyymmdd, end_yyyymmdd):
    r = requests.get(URL, params={
        "parameters": "PRECTOTCORR", "community": "AG", "longitude": lon, "latitude": lat,
        "start": start_yyyymmdd, "end": end_yyyymmdd, "format": "JSON"}, timeout=60)
    r.raise_for_status()
    daily = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    vals = {d: v for d, v in daily.items() if v is not None and v >= 0}
    return {"total_mm": round(sum(vals.values()), 1), "diario_mm": vals,
            "fonte": "NASA POWER, PRECTOTCORR (diário)"}


CLIMA = ("PRECTOTCORR", "T2M", "T2M_MAX", "RH2M", "GWETTOP", "GWETROOT")


def climate_daily(lon, lat, start_yyyymmdd, end_yyyymmdd, params=CLIMA):
    """Clima diário do NASA POWER no centro da área: chuva, temperatura, umidade do ar e umidade do
    solo modelada (GWETTOP = camada de superfície, GWETROOT = zona das raízes, 0 a 1, MERRA-2)."""
    r = requests.get(URL, params={
        "parameters": ",".join(params), "community": "AG", "longitude": lon, "latitude": lat,
        "start": start_yyyymmdd, "end": end_yyyymmdd, "format": "JSON"}, timeout=90)
    r.raise_for_status()
    p = r.json()["properties"]["parameter"]
    return {k: {d: v for d, v in p[k].items() if v is not None and v > -900} for k in params if k in p}


def antecedent_index(rain_daily, date_yyyymmdd, days=7, k=0.85):
    """Índice de chuva antecedente: soma da chuva dos dias ANTERIORES à passagem, pesando mais os recentes.
    (O NISAR passa às ~5h30 locais; a chuva do próprio dia quase sempre vem depois.)"""
    from datetime import datetime, timedelta
    d0 = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    tot = 0.0
    for i in range(1, days + 1):
        tot += rain_daily.get((d0 - timedelta(days=i)).strftime("%Y%m%d"), 0.0) * k ** (i - 1)
    return tot
