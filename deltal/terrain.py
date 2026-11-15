# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Relevo: declividade, sombra e layover a partir do DEM (Apostila de cálculo, contas 8 e 9).

A geometria de visada muda por sensor:
  Sentinel-1 olha para a DIREITA; NISAR olha para a ESQUERDA (metadados ISO: lookDirection Left).
  Órbita ascendente segue ~N (rumo ~ -10°); descendente segue ~S (rumo ~ 190°).
Sem a camada de incidência do produto, usa-se um ângulo médio (aproximação declarada na ficha).
"""
import numpy as np

HEADING = {"ascending": -10.0, "descending": 190.0, "A": -10.0, "D": 190.0}
LOOK_SIDE = {"sentinel-1": +90.0, "nisar": -90.0}   # direita = +90° do rumo, esquerda = -90°
MEAN_INCIDENCE = {"sentinel-1": 37.0, "nisar": 37.0}


def gradients(dem, pixel_m):
    dz_drow, dz_dcol = np.gradient(np.asarray(dem, dtype="float64"), pixel_m)
    return dz_dcol, -dz_drow          # (leste, norte): linhas da imagem crescem para o SUL


def slope_deg(dem, pixel_m):
    gx, gy = gradients(dem, pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def look_azimuth(sensor, orbit_state):
    return (HEADING[orbit_state] + LOOK_SIDE[sensor]) % 360.0


def radar_masks(dem, pixel_m, sensor, orbit_state, incidence_deg=None):
    """Devolve (layover, sombra, alpha_visada) para um radar e direção de órbita."""
    theta = MEAN_INCIDENCE[sensor] if incidence_deg is None else incidence_deg
    phi = np.radians(look_azimuth(sensor, orbit_state))
    gx, gy = gradients(dem, pixel_m)
    s = gx * np.sin(phi) + gy * np.cos(phi)       # >0: terreno sobe ao se afastar do satélite
    alpha = np.degrees(np.arctan(np.abs(s)))
    layover = (s > 0) & (alpha > theta)
    shadow = (s < 0) & (alpha > 90.0 - theta)
    return layover, shadow, alpha
