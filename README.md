# Hydroclimatic Whiplash in Southern California

Code and summary tables analyzing hydroclimatic whiplash events and vegetation/soil moisture responses across habitat and elevation gradients in Southern California (1982–2022).

## Repository Structure

- `src/01_detection/` — Precipitation whiplash detection scripts (`SD_0.8` and `Percentile_20_80` thresholds across `K` windows).
- `src/02_change_analysis/` — Pre/post whiplash change analysis for NDVI and root-zone soil moisture (`SMrz`), including statistical models and habitat/elevation summaries.
- `src/03_lag_correlation/` — Precipitation-leading lagged correlation analysis against vegetation, soil moisture, and climate variables.
- `src/04_figures/` — Figure generation and summary visualization scripts (`Figure_1` through `Figure_5` and lag correlation heatmaps).
- `data/` — Event catalog CSVs (`SD_0.8` whiplash events).
- `results/` — Compact statistical and summary tables.

## Pipeline Order

1. Run precipitation whiplash detection (`01_detection/`).
2. Run NDVI and `SMrz` change/event statistics (`02_change_analysis/`).
3. Run lagged correlation analysis (`03_lag_correlation/`).
4. Generate figures and summary graphics (`04_figures/`).

## External Data Note

Large raw datasets (such as full PRISM grids, GIMMS NDVI NetCDF files, GLEAM soil moisture files, shapefiles, and high-resolution GeoTIFF/PNG outputs) are kept local and excluded via `.gitignore` rather than stored in the repository.
