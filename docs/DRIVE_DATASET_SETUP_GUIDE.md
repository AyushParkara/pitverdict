# MDCE Drive Dataset Setup Guide

Use this when you are at home.

No Windows scripting needed. Use browser downloads, Google Drive upload, and Colab.

## Rule

Do not use the office PC for dataset downloads.

Use:

```text
Home laptop browser -> download zip files
Google Drive -> store all datasets
Google Colab -> extract/process/build DB
```

## Google Drive Folder Structure

Create this in Google Drive:

```text
MyDrive/
  MDCE/
    project/
    data/
      raw/
        zips/
        extracted/
      processed/
    databases/
    outputs/
      reports/
      charts/
      screenshots/
    notebooks/
    docs/
      source_pages/
```

Meaning:

- `project/`: this code project later, if you upload it to Drive.
- `data/raw/zips/`: downloaded zip files exactly as downloaded.
- `data/raw/extracted/`: extracted files in Colab.
- `data/processed/`: cleaned MDCE-ready CSV files.
- `databases/`: SQLite DB files generated in Colab.
- `outputs/`: reports, charts, screenshots.
- `docs/source_pages/`: screenshots or copied licence notes from source pages.

## Download First

Download these first. They are small enough and useful.

### 1. Kaggle Weather/Tyre Features

URL:

```text
https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
```

Save/upload as:

```text
MDCE/data/raw/zips/kaggle-naven-weather-tyre-features.zip
```

Why:

- best first real-data source for MDCE
- has lap time, tyre compound, tyre life, weather
- currently used by our adapter

Expected file:

```text
f1_all.parquet
```

Important:

- Kaggle metadata showed CC BY-NC 4.0, so confirm challenge use before final submission.

### 2. Kaggle Formula 1 Race Data

URL:

```text
https://www.kaggle.com/datasets/jtrotman/formula-1-race-data
```

Save/upload as:

```text
MDCE/data/raw/zips/kaggle-jtrotman-formula-1-race-data.zip
```

Why:

- updated Ergast-style race database
- lap times, pit stops, races, results, drivers, constructors
- useful DB foundation

Expected files:

```text
lap_times.csv
pit_stops.csv
races.csv
results.csv
drivers.csv
constructors.csv
circuits.csv
qualifying.csv
```

### 3. Kaggle World Championship Dataset

URL:

```text
https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
```

Save/upload as:

```text
MDCE/data/raw/zips/kaggle-rohanrao-f1-world-championship.zip
```

Why:

- stable historical F1 database
- good backup to compare against jtrotman

Note:

- It may be less fresh than jtrotman, but widely used.

### 4. F1DB CSV

URL:

```text
https://github.com/f1db/f1db/releases
```

Download asset:

```text
f1db-csv.zip
```

Save/upload as:

```text
MDCE/data/raw/zips/f1db-csv.zip
```

Why:

- clean metadata support
- good for race/driver/team/circuit references
- useful DB backup

### 5. Zenodo Curated F1 Telemetry Dataset

URL:

```text
https://zenodo.org/records/20061496
```

Download the zip file from the Zenodo page.

Save/upload as:

```text
MDCE/data/raw/zips/zenodo-f1-telemetry-2024.zip
```

Why:

- good curated source
- includes docs, provenance, sample outputs, data dictionary

Important:

- Do not blindly extract everything locally.
- It may contain cache files and a bundled environment.
- Extract/process in Colab only.

## Optional Later Downloads

Only download these if the first batch is not enough.

### Mendeley Safety-Car Dataset

URL:

```text
https://data.mendeley.com/datasets/djr8rnjtjp
```

Why:

- useful for safety-car uncertainty
- strong match for MDCE risk/confidence story

Before using:

- confirm exact file size
- confirm schema
- confirm licence

### Hugging Face Dataset

URL:

```text
https://huggingface.co/datasets/renumics/f1_dataset
```

Why:

- possible model-ready F1 dataset

Warning:

- around 300 MB parquet / 552 MB memory estimate
- not first priority

Avoid initially:

```text
https://huggingface.co/datasets/renumics/f1_dataset_m
```

Reason:

- around 4 GB parquet / 5.7 GB memory estimate

### TracingInsights

URL:

```text
https://tracinginsights.com/race-data/
```

Warning:

- yearly archives can be roughly 15 GB to 21 GB each
- do not download full yearly archives until MVP works

## API Sources We Do Not Download As Zip

These are not normal one-click dataset downloads. Use them later from Colab.

### FastF1

Docs:

```text
https://docs.fastf1.dev/
```

Use later for:

- lap timing
- tyre/stint info
- weather
- track status
- telemetry-style session data

Rule:

- start with one race and one or two drivers only
- do not pull a full season first

### OpenF1

Docs:

```text
https://openf1.org/docs/
```

Use later for:

- recent F1 API data
- laps, stints, pit, race-control-style endpoints depending on availability

Rule:

- start with one session
- avoid full-season high-frequency car data at first

### Open-Meteo

Docs:

```text
https://open-meteo.com/en/docs/historical-weather-api
```

Use later for:

- weather cross-checks if race weather is weak/missing

## What To Upload To Drive

Upload all downloaded zip files exactly here:

```text
MyDrive/MDCE/data/raw/zips/
```

Do not rename internal CSV/parquet files.

Do not manually edit raw files.

Do not extract on Windows if you can avoid it.

## What To Save From Source Pages

For each dataset, save proof in:

```text
MyDrive/MDCE/docs/source_pages/
```

Save either screenshots or copied notes for:

- dataset URL
- licence
- file size
- last updated date
- author/source

Minimum screenshots:

```text
kaggle-naven-weather-tyre-licence.png
kaggle-jtrotman-licence.png
kaggle-rohanrao-licence.png
f1db-release-page.png
zenodo-licence-page.png
```

## Colab Work After Upload

Open:

```text
notebooks/colab_real_data_pipeline.ipynb
```

In Colab it will:

1. mount Drive
2. create folders
3. use/upload the dataset zip
4. extract the parquet
5. prepare MDCE CSV
6. run confidence analysis
7. save a report to Drive

Expected output files:

```text
MyDrive/MDCE/data/processed/mdce_kaggle_weather_tyre_latest.csv
MyDrive/MDCE/data/processed/mdce_kaggle_weather_tyre_latest.metadata.json
MyDrive/MDCE/outputs/reports/mdce_colab_real_data_report.json
```

NonCommercial note (important):

- The Kaggle weather/tyre dataset metadata currently indicates **CC BY-NC 4.0**.
- In this repo, the default local loader is conservative: if the prepared file is detected as NonCommercial (or the sidecar metadata is missing/unreadable), MDCE will not auto-load it unless you explicitly opt in.
- Opt in for local runs by setting:

```bash
export MDCE_ALLOW_NONCOMMERCIAL_DATA=1
```

## DB Plan

After the zips are in Drive, we build one SQLite database in Colab:

```text
MyDrive/MDCE/databases/mdce_f1.db
```

It should contain:

```text
kaggle_weather_tyre_laps
jtrotman_lap_times
jtrotman_pit_stops
jtrotman_races
jtrotman_results
jtrotman_drivers
jtrotman_constructors
jtrotman_circuits
f1db_pit_stops
f1db_races
f1db_drivers
mdce_processed_laps
```

Why SQLite:

- one file
- easy in Colab
- easy to share in Drive
- no database server setup
- works on weak laptops

## Download Priority

If time/internet is limited:

1. Kaggle weather/tyre features
2. Kaggle jtrotman race data
3. F1DB CSV
4. Zenodo
5. Kaggle rohanrao backup
6. Mendeley safety-car dataset
7. FastF1/OpenF1 one-session pulls in Colab

Do not start with:

- multi-GB Hugging Face dataset
- TracingInsights yearly archives
- commercial APIs

## Final Reminder

Raw data belongs in Drive.

Code belongs in project/GitHub.

Generated reports/screenshots belong in Drive outputs.

For final submission, every dataset must have:

- URL
- licence
- exact file used
- fields used
- derived/proxy field explanation
