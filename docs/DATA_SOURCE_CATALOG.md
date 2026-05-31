# MDCE Data Source Catalog

Purpose:

> Collect links and references for possible MDCE data sources. Do not download now. Download/export later into Google Drive.

Related:

- Dataset size/download priority matrix: `DATASET_SIZE_MATRIX.md`

## Best Primary Sources

### 1. FastF1

Link:

https://docs.fastf1.dev/

Useful docs:

- Core timing/telemetry: https://docs.fastf1.dev/core.html
- Data reference: https://docs.fastf1.dev/data_reference/index.html
- Session object: https://docs.fastf1.dev/api_reference/session.html
- Jolpica/Ergast interface: https://docs.fastf1.dev/api_reference/jolpica.html

What it can provide:

- lap timing
- sector times
- tyre compound
- tyre life/stint data
- pit-related lap info
- car telemetry such as speed, RPM, gear, throttle/brake where available
- weather data
- track status
- race control messages

MDCE usefulness:

Very high.

Best for:

- lap-time degradation
- model-vs-reality comparison
- safety-car/track-status context
- weather-aware confidence
- real telemetry-derived indicators

Notes:

- This should be one of the first sources to test in Colab.
- Some sessions/years may have incomplete telemetry/weather availability.

### 2. OpenF1

Link:

https://openf1.org/docs/

What it can provide:

- historical F1 data from 2023 onward
- JSON/CSV API access
- lap data
- car data at around 3.7 Hz
- sessions
- drivers
- stints
- pit data
- intervals
- race control/team radio/weather-style endpoints depending on endpoint availability

MDCE usefulness:

Very high for recent races.

Best for:

- recent-season lap and telemetry workflows
- lightweight API access in Colab
- CSV/JSON exports

Notes:

- Historical data is accessible without authentication according to the docs.
- Real-time data may require paid subscription.
- Unofficial and not associated with Formula 1.

### 3. Jolpica-F1 / Ergast Successor

Link:

https://docs.fastf1.dev/api_reference/jolpica.html

Related historical API idea:

https://ergast.com/mrd/

What it can provide:

- historical results
- schedules
- drivers
- constructors
- lap times
- pit stops
- qualifying
- standings

MDCE usefulness:

Medium-high.

Best for:

- historical context
- pit-stop timing
- lap-time tables
- driver/team metadata

Limitations:

- Not high-frequency telemetry.
- Better for historical strategy context than detailed telemetry.

## Curated Dataset Sources

### 4. Kaggle: Formula 1 Race Data / World Championship Dataset

Useful links:

- https://www.kaggle.com/datasets/jtrotman/formula-1-race-data
- https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

What it can provide:

- races
- drivers
- constructors
- lap times
- pit stops
- qualifying
- results
- standings

MDCE usefulness:

Medium.

Best for:

- historical lap/pit context
- basic race strategy examples
- joins with other datasets

Limitations:

- Usually not telemetry-level.
- Check exact license on Kaggle before submission.

### 5. Kaggle: Formula 1 Dataset With Weather & Tyre Features

Link:

https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features

What it can provide:

- lap-level race data
- weather features
- tyre/stint features
- Ergast/FastF1-derived processed features

MDCE usefulness:

High if license allows challenge use.

Best for:

- immediate model-ready Colab work
- lap-time prediction baseline
- tyre/weather confidence scenarios

License note:

Search result shows CC BY-NC 4.0. Confirm on Kaggle before using in final submission.

### 6. Zenodo: Curated Formula 1 Telemetry and Race Session Dataset, 2024 Full Season

Link:

https://zenodo.org/records/20061496

What it can provide:

- 2024 full-season session/race dataset
- FastF1/Jolpica-derived analysis-ready data
- lap timing
- sector splits
- tyre compounds
- pit indicators
- weather snapshots
- driver/team metadata

MDCE usefulness:

Very high if accessible and license is acceptable.

Best for:

- one clean real-data MVP without building all collection scripts first
- 2024 season examples

Need to check:

- exact license
- file size
- schema
- whether telemetry is lap-level or high-frequency

### 7. Mendeley: Multi-Season F1 Lap Dataset With Safety Car Labels

Link:

https://data.mendeley.com/datasets/djr8rnjtjp

What it can provide:

- 2022-2025 lap-level race data
- safety-car labels from race-control context
- forward-looking safety-car target possibilities

MDCE usefulness:

Very high for uncertainty and safety-car scenarios.

Best for:

- safety-car context confidence
- strategy-risk explanation
- "decision trust changes under race context" demo

Need to check:

- license
- exact columns
- whether it includes tyre/stint/pit fields

## Historical/Open Databases

### 8. F1DB

Link:

https://github.com/f1db/f1db

Releases:

https://github.com/f1db/f1db/releases

What it can provide:

- open source Formula 1 database
- CSV/JSON/SQL-style exports
- all-time F1 racing data/statistics

MDCE usefulness:

Medium.

Best for:

- driver/team/race context
- clean metadata joins
- historical references

Limitations:

- Not lap telemetry.

### 9. Formula 1 Archive

Link:

https://www.formula1archive.com/

About/data methodology:

https://www.formula1archive.com/about

What it can provide:

- historical F1 database
- results
- qualifying
- lap times
- pit stops
- weather data
- safety-car deployments

MDCE usefulness:

Medium-high if data can be exported/queried cleanly.

Best for:

- safety-car/weather context
- historical race facts

Need to check:

- data export/API availability
- terms of use

### 10. CTU Relational ErgastF1

Link:

https://relational.fel.cvut.cz/dataset/ErgastF1

What it can provide:

- relational version of Ergast-style F1 data
- lap times
- pit stops
- qualifying
- results

MDCE usefulness:

Medium.

Best for:

- relational ML experiments
- historical context

Limitations:

- Older coverage in search result: 1950-2017.
- Not telemetry-level.

## Weather Support Sources

### 11. Open-Meteo Historical Weather API

Link:

https://open-meteo.com/en/docs/historical-weather-api

What it can provide:

- historical weather by latitude/longitude/date
- temperature
- humidity
- precipitation/rain
- wind speed/direction
- pressure/cloud variables depending on request

MDCE usefulness:

High as support data.

Best for:

- augmenting race sessions with independent weather context
- weather uncertainty scenarios

Limitations:

- Circuit-local weather may differ from exact track sensor readings.
- Must align timestamps and circuit coordinates carefully.

### 12. Meteostat

Link:

https://dev.meteostat.net/

What it can provide:

- historical weather/climate data
- JSON API
- Python library

MDCE usefulness:

Medium-high as support data.

Best for:

- cross-checking weather context
- independent weather source

## Optional/Commercial/Needs Caution

### 13. PitStop Data API

Link:

https://pitstopdata.com/

What it can provide:

- lap times
- sector splits
- tyre compounds
- pit stop strategy data
- standings/circuit metadata

MDCE usefulness:

Potentially high.

Need to check:

- API key
- RapidAPI terms
- free-tier limits
- whether challenge use is allowed

### 14. T1API

Link:

https://docs.t1f1.com/

What it can provide:

- telemetry API
- historical/live F1 telemetry-style endpoints

MDCE usefulness:

Potentially high.

Need to check:

- pricing
- terms
- source/redistribution rights

### 15. RacingHub API

Link:

https://racinghub.net/api/v1/

What it can provide:

- historical F1 data/statistics API
- race results
- qualifying
- sprint
- pit stop data
- standings

MDCE usefulness:

Medium.

Need to check:

- terms
- stability
- field coverage

### 16. f1api.dev

Link:

https://f1api.dev/

What it can provide:

- Formula 1 data API
- docs/GitHub available from site

MDCE usefulness:

Medium.

Need to check:

- fields
- license
- current availability

## Additional Backup Sources

These are backup or enrichment sources. Use them only if the primary path is insufficient.

### 17. FIA Results & Statistics Documents

Link:

https://www.fia.com/events/fia-formula-one-world-championship/season-2025/2025-fia-formula-one-world-championship-results

Example event/session document pages often include:

- final classification PDFs
- lap chart PDFs
- pit stop summary PDFs
- race history chart PDFs
- sector/time-related official documents depending on event

MDCE usefulness:

High as an official fallback for:

- pit stops
- lap charts
- classifications
- race-control/session facts

Limitations:

- Mostly PDFs, so extraction is more work.
- Need parsing/cleaning.
- Coverage depends on event pages and document availability.
- Check FIA terms before redistribution.

Possible helper:

- Docling can parse PDFs and would also satisfy IBM technology usage if used carefully.

### 18. Formula1.com Results Pages

Link:

https://www.formula1.com/en/results

What it can provide:

- race results
- qualifying results
- practice results
- fastest laps
- pit stop summaries in some result sections/seasons

MDCE usefulness:

Medium.

Best for:

- official-looking verification
- simple result/pit context
- cross-checking other datasets

Limitations:

- Not a telemetry source.
- Scraping terms must be checked.
- Better as reference/cross-check, not primary data.

### 19. TracingInsights Race Data

Link:

https://tracinginsights.com/race-data/

What it can provide:

- Formula 1 timing/analysis data exports
- lap/stint/race data depending on available files

MDCE usefulness:

Medium-high if usable/exportable.

Best for:

- quick public race analysis data
- backup when FastF1/OpenF1 flow is slow

Need to check:

- exact downloadable formats
- licence/usage permissions
- whether data source is FastF1-derived

### 20. Hugging Face Datasets

Search link:

https://huggingface.co/datasets?search=formula%201

Possible useful searches:

```text
Formula 1 telemetry
Formula 1 lap times
F1 race data
FastF1
```

MDCE usefulness:

Medium.

Best for:

- quick experimental datasets
- model-ready tables
- fallback samples for Colab

Need to check:

- dataset card
- licence
- fields
- source provenance
- whether it is duplicate FastF1/Ergast data

### 21. Data.World Formula 1 Datasets

Search link:

https://data.world/search?q=formula%201&type=resources

MDCE usefulness:

Medium.

Best for:

- historical F1 tables
- possible Ergast mirrors
- basic lap/pit datasets

Need to check:

- login/download access
- licence
- freshness
- whether it contains lap-level data or only results

### 22. GitHub Dataset Mirrors

Search links:

```text
https://github.com/search?q=formula+1+lap+times+csv&type=repositories
https://github.com/search?q=f1+pit+stops+csv&type=repositories
https://github.com/search?q=fastf1+csv+dataset&type=repositories
https://github.com/search?q=formula+1+weather+tyre+dataset&type=repositories
```

MDCE usefulness:

Medium as backup.

Best for:

- small CSVs
- notebooks with processed FastF1 data
- examples of feature engineering

Need to check:

- licence file
- commit freshness
- whether data is copied from Kaggle/Ergast/FastF1
- whether redistribution is allowed

Rule:

Do not use random GitHub CSVs in the final project unless provenance and licence are clear.

### 23. FIA PDF Parser / Extraction Tools

Search link:

https://github.com/search?q=FIA+Formula+1+PDF+lap+chart+parser&type=repositories

Potential use:

- parse FIA lap charts
- parse pit stop summaries
- parse official timing PDFs

MDCE usefulness:

Medium as a backup path if official PDFs are the chosen source.

Need to check:

- parser accuracy
- maintenance status
- licence
- whether it supports current FIA document formats

### 24. SportsDataIO Formula 1 API

Link:

https://sportsdata.io/developers/api-documentation/formula-1

MDCE usefulness:

Medium.

Best for:

- structured paid API fallback
- schedules/results/standings/race metadata

Need to check:

- pricing
- data fields
- whether lap/pit/telemetry data exists
- challenge-use permission

### 25. Sportradar Motorsports API

Link:

https://developer.sportradar.com/

MDCE usefulness:

Medium-low for MVP because it is commercial.

Best for:

- professional structured sports data if available through access

Need to check:

- API access
- pricing
- F1 coverage depth
- redistribution rights

### 26. Motorsport Magazine / StatsF1 / Racing-Reference Style Archives

Examples:

- https://www.statsf1.com/
- https://www.motorsportmagazine.com/database/championships/f1/
- https://www.racing-reference.info/

MDCE usefulness:

Low-medium.

Best for:

- historical cross-checks
- race facts
- context in README

Limitations:

- Usually not downloadable structured telemetry.
- Scraping may be disallowed.
- Not ideal as primary data.

## Research Papers / Modeling References

These are not necessarily datasets, but they support the MDCE logic.

### 27. State-Space Tire Degradation In F1

Link:

https://arxiv.org/abs/2512.00640

Use:

- supports uncertainty-aware tyre degradation modeling
- useful for README/methodology

### 28. Explainable Time Series Prediction of Tyre Energy

Link:

https://arxiv.org/abs/2501.04067

Use:

- shows tyre energy/telemetry is a real race-strategy modeling problem
- also reminds us private team telemetry is not publicly available

## Recommended Data Plan For MDCE

### Best MVP Path

Use:

1. FastF1 or OpenF1 for real lap/session data.
2. Mendeley safety-car-labelled dataset or FastF1 track status for safety-car context.
3. Open-Meteo only if weather fields are missing or weak.

### Best Backup Path

Use:

1. Kaggle weather/tyre features dataset.
2. Kaggle/F1DB/Jolpica historical pit/lap data.
3. FIA PDFs if official pit/lap charts are needed.
4. Open-Meteo/Meteostat if weather fields are missing.

### Last-Resort Backup Path

Use:

1. Formula1.com/FIA result pages for official context.
2. GitHub/Hugging Face/Data.World mirrors only after licence/provenance checks.
3. Commercial APIs only if access is available and terms allow challenge use.

### Avoid For MVP

Avoid relying only on:

- generic historical results
- championship standings
- driver/team metadata

Those are useful context but not enough for MDCE confidence scoring.

## Data Selection Rule

Pick data that supports at least four of these:

- lap time
- sector times
- tyre compound
- tyre age/stint
- pit stop laps
- track status/safety car
- weather
- car telemetry or speed
- model prediction target

If a source does not support at least four, it is probably not enough as the primary MDCE dataset.

## Priority Ladder

Use this order when choosing data:

1. **FastF1 or OpenF1**: best real-data path for lap/session/telemetry-style work.
2. **Mendeley safety-car dataset / Zenodo curated dataset**: best ready-made backup if licence is acceptable.
3. **Kaggle tyre/weather/lap datasets**: useful if schemas are already clean.
4. **Jolpica/Ergast/F1DB**: reliable historical context but less telemetry depth.
5. **Open-Meteo/Meteostat**: weather enrichment, not primary race data.
6. **FIA PDFs / Formula1.com result pages**: official backup/cross-check, but parsing can take time.
7. **GitHub/Hugging Face/Data.World mirrors**: use only after provenance/licence checks.
8. **Commercial APIs**: only if free access and terms fit the challenge.
