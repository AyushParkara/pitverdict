# Data Provenance

## Current Local Data

When this file exists, the app now uses real public prepared data by default:

```text
data/processed/mdce_kaggle_weather_tyre_latest.csv
```

Current prepared source:

- Kaggle Formula 1 Dataset With Weather and Tyre Features
- 2023 round 22
- Yas Marina Circuit
- driver code: VER
- decision lap: 16
- rows: 16

Licence status (must confirm before final submission):

- Kaggle dataset license is **CC BY-NC 4.0** (exposed in the Kaggle dataset page metadata).
- This means the dataset is restricted for commercial use. We must re-check Kaggle terms and ensure the challenge/demo use is compliant before any final/public submission.

NonCommercial guardrail (local default behavior):

- When the prepared dataset is detected as NonCommercial (e.g. CC BY-NC), MDCE will **not** auto-load it by default.
- To allow using it as the default local dataset, set:

```bash
export MDCE_ALLOW_NONCOMMERCIAL_DATA=1
```

- You can always bypass this by uploading your own processed MDCE CSV in the app sidebar.

The app still has an offline fallback dataset:

```text
data/sample_race.csv
```

This fallback is only for local prototype testing when processed real data is unavailable.

## Why Fallback Data Exists

The project needs to run locally even when:

- internet is unavailable
- FastF1 download/cache setup fails
- private telemetry is not available
- IBM/API access is not configured

So the repo includes a controlled fallback dataset to prove the MDCE logic.

## Real Data Workflow

The main project workflow uses actual public data prepared into the MDCE schema:

```text
raw data -> mapping/cleaning -> processed MDCE CSV -> app default/upload
```

Current local preparation command:

```bash
.venv/bin/python tools/prepare_mdce_dataset.py
```

Google Drive/Colab workflow can still use:

```text
notebooks/colab_drive_workflow.ipynb
```

## What The Dataset Contains

The prepared CSV contains real/source fields:

- lap number
- lap time from `milliseconds`
- tyre compound
- tyre life/age
- rainfall
- track temperature
- air temperature
- humidity
- pressure
- wind speed
- source year/round/race/circuit/driver metadata

MDCE then derives or proxies missing strategy fields:

- sector times are derived from lap time
- weather is derived from rainfall
- track status defaults to `NORMAL` unless another source is joined
- gap to car ahead defaults to a neutral fallback
- `predicted_lap_time_s` is derived from prior rolling lap-time average
- `tyre_temp_proxy_c` is a public track-temperature-based proxy
- `speed_consistency` is a lap-time stability proxy

The offline fallback dataset also contains synthetic proxy fields:

- `predicted_lap_time_s`
- `tyre_temp_proxy_c`
- `speed_consistency`

These are intentionally synthetic and exist only to demonstrate uncertainty/conflict scenarios.

## Other Downloaded Sources (Not Used By Default Yet)

This repo also contains locally downloaded archives under `data/raw/` (ignored by git) that can be used later to expand coverage.

See:

```text
DATA_DOWNLOAD_MANIFEST.md
DATASET_SIZE_MATRIX.md
DATA_SOURCE_CATALOG.md
```

Known licence posture based on what is available locally today:

- Zenodo curated telemetry bundle (selected extract):
  - Source: https://zenodo.org/records/20061496
  - Local licence files:
    - `data/raw/extracted/zenodo_2024_selected/anandm84-F1-telemetry-DE-592e162/LICENSE` (MIT for code only)
    - `data/raw/extracted/zenodo_2024_selected/anandm84-F1-telemetry-DE-592e162/LICENSE-DATA.md` (CC-BY-4.0 for derived data + docs)
  - Important: the Zenodo project explicitly states it does **not** redistribute raw F1 timing feeds; it provides derived outputs + reproducible extraction code.

- Kaggle Ergast-style datasets (jtrotman/rohanrao):
  - License was recorded as **CC0** in `DATASET_SIZE_MATRIX.md` at check time.
  - Still re-check Kaggle pages before final submission.

- F1DB release asset (`f1db-csv.zip`):
  - Upstream repo: https://github.com/f1db/f1db
  - Licence appears to be **CC-BY-4.0** (per upstream `LICENSE` in the repository).
  - Still confirm before final submission, and ensure attribution is included where required.

## Attribution Note (If We Use Zenodo Derived Data)

If the Zenodo-derived curated tables or docs are used in the demo/submission, we must include an attribution note. The bundled `LICENSE-DATA.md` provides the required boilerplate.

## What We Must Not Claim

Do not claim:

- this is real F1 team telemetry
- this is private tyre-temperature data
- this is an official FIA/F1 dataset
- this represents exact real car behavior

## What We Can Claim

Safe claim:

> MDCE can ingest public real lap/tyre/weather data mapped into its schema, then clearly separates real/source fields from derived and proxy fields before producing strategy confidence.

## Future Real Data Option

The next realistic upgrade is to connect:

```text
FastF1
```

FastF1 can provide public historical race/session data such as:

- lap times
- sector times
- tyre compound
- stint information
- timing/session data

FastF1 docs:

https://docs.fastf1.dev/

Even with FastF1, some fields like tyre temperature and private team model outputs are not publicly available, so those must remain labelled as demo proxies unless replaced by a legitimate source.
