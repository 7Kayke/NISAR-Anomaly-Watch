# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""As três camadas de mudança, calculadas sempre juntas.

  vegetacao : HV (ou VH) cai forte e cai mais que HH (ou VV)  -> fogo, desmatamento
  agua      : co-pol (HH ou VV) cai forte e fica escuro       -> água lisa, cheia
  dano      : coerência cai (GUNW)                            -> superfície destruída ou mexida

Os limiares ficam em PARAMS e devem ser calibrados no local de treino e congelados.
"""
import numpy as np

from .change import log_ratio_db, robust_z, min_area_filter, to_db, boxcar_linear

PARAMS = {
    "filter_size": 5,
    "z_k": 2.5,             # |z| mínimo
    "veg_drop_db": -2.0,    # queda mínima do cross-pol
    "veg_margin_db": -1.0,  # cross-pol cai pelo menos 1 dB a mais que o co-pol
    "water_drop_db": -3.0,  # queda mínima do co-pol
    "water_dark_db": -18.0, # co-pol depois abaixo disso
    "coh_drop": 0.3,        # queda mínima de coerência
    "min_area_m2": 4000.0,  # 0,4 ha
}


def compute_layers(co_before, co_after, cross_before, cross_after, pixel_area_m2,
                   valid=None, coh_before=None, coh_event=None, params=None):
    p = dict(PARAMS, **(params or {}))
    k = p["filter_size"]
    lr_co = log_ratio_db(co_before, co_after, k)
    lr_x = log_ratio_db(cross_before, cross_after, k)
    if valid is None:
        valid = np.isfinite(lr_co) & np.isfinite(lr_x)
    z_co, med_co, mad_co = robust_z(lr_co, valid)
    z_x, med_x, mad_x = robust_z(lr_x, valid)
    co_after_db = to_db(boxcar_linear(co_after, k))

    veg = valid & (z_x < -p["z_k"]) & (lr_x < p["veg_drop_db"]) & ((lr_x - lr_co) < p["veg_margin_db"])
    water = valid & (z_co < -p["z_k"]) & (lr_co < p["water_drop_db"]) & (co_after_db < p["water_dark_db"])
    min_px = max(1, int(round(p["min_area_m2"] / pixel_area_m2)))
    veg, veg_sizes = min_area_filter(veg, min_px)
    water, water_sizes = min_area_filter(water, min_px)

    out = {
        "lr_co": lr_co, "lr_cross": lr_x, "z_co": z_co, "z_cross": z_x,
        "vegetacao": veg, "agua": water,
        "stats": {
            "lr_co_mediana_db": med_co, "lr_co_mad_db": mad_co,
            "lr_cross_mediana_db": med_x, "lr_cross_mad_db": mad_x,
            "vegetacao_area_ha": float(veg.sum() * pixel_area_m2 / 1e4),
            "agua_area_ha": float(water.sum() * pixel_area_m2 / 1e4),
            "vegetacao_grupos": veg_sizes, "agua_grupos": water_sizes,
            "pixels_validos": int(valid.sum()),
        },
        "params": p,
    }
    if coh_before is not None and coh_event is not None:
        d = np.asarray(coh_before) - np.asarray(coh_event)
        dano = valid & np.isfinite(d) & (d > p["coh_drop"])
        dano, dano_sizes = min_area_filter(dano, min_px)
        out["dano"] = dano
        out["delta_coerencia"] = d
        out["stats"]["dano_area_ha"] = float(dano.sum() * pixel_area_m2 / 1e4)
        out["stats"]["dano_grupos"] = dano_sizes
    return out
