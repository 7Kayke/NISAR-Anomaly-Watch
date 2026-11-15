# NISAR Anomaly Watch (team Delta L)

NASA Space Apps Challenge 2026, challenge "Dancing with the SARs".

Pick any area and two dates. The pipeline finds the NISAR scenes, reads only the
area of interest, computes surface change in three layers (vegetation loss,
new water, damage/motion), cross-checks it with other satellites, validates it
against independent ground truth and exports a technical sheet.

## What runs

| Step | Data | Access |
| --- | --- | --- |
| Ground truth for fires | Sentinel-2 L2A, dNBR | Microsoft Planetary Computer (STAC, no account) |
| Second radar | Sentinel-1 RTC (VV, VH) | Microsoft Planetary Computer |
| Main radar | NISAR L2 GCOV (HH, HV), GUNW | NASA CMR + earthaccess (Earthdata Login) |
| Context | Daily rainfall (PRECTOTCORR) | NASA POWER API |

Method (see `deltal/`): linear power -> 5x5 boxcar -> log-ratio in dB ->
spatial robust z-score (median, 1.4826 x MAD) -> thresholds -> minimum area ->
three layers -> confusion matrix (precision, recall, F1, IoU, false-alarm rate).

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # synthetic tests
python run_case.py --case cases/norte_fire.json --out docs/results/norte_fire
```

NISAR needs `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (environment variables or
`~/.netrc`). Without them, add `--skip-nisar`.

On GitHub: add both as repository Secrets, then Actions -> "Analisar caso" ->
Run workflow. Results are committed to `docs/results/<case>/`.

## Outputs

`ficha_tecnica.json` / `.csv` (every file, date, geometry, parameter and metric),
`resultado.md` (plain-language summary), PNG maps.

## AI use

| What | Who |
|---|---|
| Project idea, study area, choice of sensors and data sources, collection funnel concept, review of every result | Team Delta L |
| Much of the Python code, its comments and docs drafts | Written with an AI assistant (Claude, Anthropic), reviewed and run by the team |
| Numbers, maps and metrics | Computed by the pipeline from real NASA / ESA / USGS data, never written by AI |

Every Python file starts with an `# AI-assisted` note, and commits made with the assistant carry a `Co-Authored-By: Claude` line.
No AI-generated images or video are used; if any are added to the demo they will carry a visible "AI-generated" label.

## References

NASA/JPL NISAR (ASF DAAC); ESA Copernicus Sentinel-1 and Sentinel-2 via Microsoft
Planetary Computer; NASA POWER; CAL FIRE incident reports. Libraries: numpy,
scipy, matplotlib, pyproj, rasterio, h5py, pystac-client, planetary-computer,
odc-stac, earthaccess.


## Unified collection funnel (multi-sensor)

`run_funnel.py` searches every source for the same area and event date and records how many items survive each stage:
`catalog → covers the area (≥90 %) → quality (cloud inside the area ≤20 %, thermal band present, VV+VH, same orbit) → paired (before/after, same geometry, smallest gap)`.
Per-stage counts and yield percentages are written to `funil.json` and the technical sheet.

| Role | Sources |
|---|---|
| Evidence (votes) | NISAR GCOV (L-band HV), Sentinel-1 RTC (C-band VH, calibrated), Landsat 8/9 C2 L2 dNBR, Landsat ΔLST (relative surface temperature) |
| Ground truth (does not vote) | Sentinel-2 L2A dNBR > 0.1 |
| Mask | Copernicus DEM 30 m: radar layover and shadow per sensor look direction (Sentinel-1 right-looking, NISAR left-looking) |
| Context | NAIP aerial photo (before), ASTER L1T, NASA POWER rainfall |

A pixel enters the consensus when at least 2 available sources detect change; confidence = votes / sources that could see the pixel.
The consensus is validated against the Sentinel-2 scar (precision, recall, F1, IoU, false-alarm rate).

```bash
python run_funnel.py --case cases/norte_fire.json --out docs/results/norte_fire_funil
```

## REST API

Local (computer A): `pip install -r requirements-api.txt && uvicorn api.app:app --port 8000`, then open `http://localhost:8000/docs`.

| Method | Path | Does |
|---|---|---|
| GET | `/v1/collections` | sources and what each measures |
| GET | `/v1/search?bbox=w,s,e,n&date=YYYY-MM-DD` | runs only the funnel (no pixel download) |
| POST | `/v1/analyses` | starts a full analysis in the background |
| GET | `/v1/analyses/{id}` | status and technical sheet |
| GET | `/v1/analyses/{id}/files/{name}` | figures, JSON, CSV |

End users without a server read the same responses as static JSON from GitHub Pages: `docs/api/v1/collections.json` and `docs/api/v1/analyses/{id}.json`.

## The waltz: full NISAR time series for a study area

`run_series.py` takes every NISAR GCOV pass over a study area (same track, direction and frame), keeps the catalog
metadata of **all** granules with the funnel stage where each one stopped (`metadados.csv`), and classifies the rhythm
of every pixel: stable, strong seasonal rhythm, abrupt drop, abrupt rise, transient flooding, permanent water.
Each date is normalized by the scene median, so the landscape's own rhythm (season, rain, calibration) is shown as a
separate curve next to NASA POWER rainfall, and the maps show only local departures.

Outputs: `classes.png`, `quando_mudou.png` (date of the abrupt change), `rgb_temporal.png` (3-period HV composite),
`curvas.png`, `animacao_hv.gif`, `serie.json` + `serie_hv_u8.bin` (per-pixel series for the interactive view).

```bash
python run_series.py --case cases/norte_fire.json --out docs/results/norte_fire_serie
```
