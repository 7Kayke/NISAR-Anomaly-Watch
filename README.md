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

Parts of the code and text were written with help from an AI assistant (Claude)
and reviewed by the team. The project idea and all decisions are the team's.

## References

NASA/JPL NISAR (ASF DAAC); ESA Copernicus Sentinel-1 and Sentinel-2 via Microsoft
Planetary Computer; NASA POWER; CAL FIRE incident reports. Libraries: numpy,
scipy, matplotlib, pyproj, rasterio, h5py, pystac-client, planetary-computer,
odc-stac, earthaccess.
