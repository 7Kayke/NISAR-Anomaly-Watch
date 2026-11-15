# AI-assisted: parts of this file were written with help from Claude (Anthropic). Reviewed and run by team Delta L.
"""Pacote de dados do quadrante, um formato para cada público.

  GIS (QGIS/ArcGIS)        COG GeoTIFF: série HH e HV (uma banda por data) e os mapas da análise
  ciência (Python/R)       cubo NetCDF (tempo, y, x) com HH, HV e os mapas
  web e decisão            GeoJSON com os polígonos de mudança (classe, data, área, intensidade)
  planilha / público       CSV com a curva da área por data
Tudo na mesma grade (UTM, resolução da análise) e com as unidades no LEIAME.
"""
import csv
import json
from pathlib import Path

import numpy as np


def _cog(path, bands, grid, names, dtype="float32", nodata=np.nan):
    import rasterio
    arr = np.asarray(bands, dtype=dtype)
    prof = {"driver": "COG", "width": grid.width, "height": grid.height, "count": arr.shape[0], "dtype": dtype,
            "crs": grid.crs, "transform": grid.transform, "nodata": nodata, "compress": "DEFLATE"}
    try:
        dst = rasterio.open(path, "w", **prof)
    except Exception:                                # GDAL antigo sem o driver COG
        prof.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256)
        dst = rasterio.open(path, "w", **prof)
    with dst:
        dst.write(arr)
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, str(n))


def change_polygons(cls, when, delta, dates, grid, S, classes=(3, 4, 5, 7, 8), min_pixels=4):
    import rasterio.features
    from rasterio.warp import transform_geom
    from scipy import ndimage
    feats = []
    px = grid.pixel_area_m2
    for c in classes:
        lab, n = ndimage.label(cls == c)
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= min_pixels)[0]
        keep = keep[keep > 0]
        if not keep.size:
            continue
        lab2 = np.where(np.isin(lab, keep), lab, 0).astype("int32")
        for geom, v in rasterio.features.shapes(lab2, mask=lab2 > 0, transform=grid.transform):
            m = lab2 == int(v)
            k = when[m]; k = k[k > 0]
            props = {"classe": S.CLASSES[c], "codigo": int(c), "area_ha": round(float(m.sum() * px / 1e4), 2),
                     "intensidade_db": round(float(np.nanmedian(delta[m])), 2) if c in (3, 4) else None}
            if k.size:
                kk = int(np.bincount(k).argmax())
                props["mudou_entre"] = [dates[kk - 1], dates[kk]]
            feats.append({"type": "Feature", "properties": props,
                          "geometry": transform_geom(grid.crs, "EPSG:4326", geom, precision=6)})
    feats.sort(key=lambda f: -f["properties"]["area_ha"])
    return {"type": "FeatureCollection", "features": feats}


def package(out, grid, dates, hh, hv, res, inv, S):
    d = Path(out) / "dados"
    d.mkdir(parents=True, exist_ok=True)
    tag = [x.replace("-", "") for x in dates]
    _cog(d / "hh_db_serie.tif", hh, grid, tag)
    _cog(d / "hv_db_serie.tif", hv, grid, tag)
    k = res["quebra_indice"].astype("float32"); k[k < 1] = np.nan
    maps = {"classe": res["classe"].astype("float32"), "quebra_indice_data": k, "quebra_delta_db": res["delta_db"],
            "tendencia_db_mes": res["tendencia_db_mes"], "ritmo_residuo_db": res["residuo_db"]}
    if inv:
        maps["r_umidade_hh"], maps["r_umidade_hv"] = inv["HH"][0], inv["HV"][0]
    _cog(d / "mapas.tif", list(maps.values()), grid, list(maps))
    geo = change_polygons(res["classe"], res["quebra_indice"], res["delta_db"], dates, grid, S)
    (d / "mudancas.geojson").write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    with open(d / "serie_area.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["data", "hh_mediana_db", "hv_mediana_db"])
        for dt, a, b in zip(dates, res["curva_cena_hh"], res["curva_cena_hv"]):
            w.writerow([dt, round(float(a), 3), round(float(b), 3)])
    try:
        import pandas as pd
        import xarray as xr
        xs = grid.transform.c + grid.transform.a * (np.arange(grid.width) + 0.5)
        ys = grid.transform.f + grid.transform.e * (np.arange(grid.height) + 0.5)
        ds = xr.Dataset({"hh_db": (("tempo", "y", "x"), hh.astype("float32")),
                         "hv_db": (("tempo", "y", "x"), hv.astype("float32")),
                         **{k: (("y", "x"), np.asarray(v, "float32")) for k, v in maps.items()}},
                        coords={"tempo": pd.to_datetime(dates), "x": xs, "y": ys})
        ds.attrs.update({"crs": str(grid.crs), "fonte": "NISAR L2 GCOV (banda L), NASA/JPL via ASF",
                         "classes": json.dumps(S.CLASSES, ensure_ascii=False), "projeto": "Delta L"})
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        try:
            ds.to_netcdf(d / "cubo.nc", engine="h5netcdf", encoding=enc)
        except Exception:
            ds.to_netcdf(d / "cubo.nc")
    except Exception as e:
        (d / "cubo_erro.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
    (d / "LEIAME.txt").write_text(f"""Pacote de dados Delta L
Grade: {grid.crs}, {grid.resolution} m, {grid.width} x {grid.height} pixels. Datas: {', '.join(dates)}

hh_db_serie.tif / hv_db_serie.tif  retroespalhamento NISAR GCOV em dB, uma banda por data (descrição = AAAAMMDD)
mapas.tif     bandas: {', '.join(maps)}
              classe: {json.dumps(S.CLASSES, ensure_ascii=False)}
              quebra_indice_data: índice da primeira data DEPOIS da mudança abrupta (1 = 2a data)
              quebra_delta_db: tamanho do degrau no HV (dB, relativo à área); tendencia_db_mes: inclinação do HV
              r_umidade_*: correlação do radar com a chuva antecedente (NASA POWER)
cubo.nc       tudo acima num cubo NetCDF (tempo, y, x), para xarray / R / Panoply
mudancas.geojson  polígonos das mudanças (EPSG:4326) com classe, área em ha, intensidade e entre quais datas
serie_area.csv    mediana do HH e do HV da área em cada data
""", encoding="utf-8")
    return d
