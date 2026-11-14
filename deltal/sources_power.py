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
