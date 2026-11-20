# ============================================================
# FIGURE 4: HABITAT + STUDY-AREA COMPOSITE TIME SERIES
#
# Outputs:
#   1. K-specific real-month composites:
#      StartDate - 3 months through EndDate + 3 months
#
#   2. All-K landmark composite, previous version:
#      Start -3, Start -2, Start -1, Start, Start +1,
#      GAP,
#      End -1, End, End +1, End +2, End +3
#
#   3. All-K landmark composite, expanded middle version:
#      Start -3, Start -2, Start -1, Start, Start +1, Start +2,
#      GAP,
#      End -2, End -1, End, End +1, End +2, End +3
#
# Important update:
#   - Landmark lines are NOT connected across the gap.
#   - Start-side and End-side landmark segments are plotted separately.
#   - A visual x-axis gap is added between the Start-side and End-side points.
#   - Keeps 9th subplot: Study Area.
# ============================================================

import os
import glob
import re
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt

from rasterio.transform import xy
from pyproj import Geod

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL GDAL / PROJ PATHS
# ============================================================

os.environ.setdefault(
    "GDAL_DATA",
    r"C:\Users\Owner\anaconda3\envs\myenv\Library\share\gdal"
)

os.environ.setdefault(
    "PROJ_LIB",
    r"C:\Users\Owner\anaconda3\envs\myenv\Library\share\proj"
)


# ============================================================
# PATHS
# ============================================================

BASE_OUT = Path(r"D:\Graduate Research\Final_Paper_Analysis")

FIGURE4_OUT_DIR = BASE_OUT / "Figure_4"
FIGURE4_FIG_DIR = FIGURE4_OUT_DIR / "Figures"
FIGURE4_CSV_DIR = FIGURE4_OUT_DIR / "CSV"

FIGURE4_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE4_FIG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE4_CSV_DIR.mkdir(parents=True, exist_ok=True)

WHIPLASH_ROOT = Path(
    r"D:\Graduate Research\Final_Paper_Analysis\Whiplash Analysis Precipitation 1982_2022"
)

WHIPLASH_EVENT_FILES = {
    "Original_MultiK": {
        3: WHIPLASH_ROOT / "Original_MultiK" / "SD_0.8" / "Whiplash_Events_Original_MultiK_SD_0.8_K3_1-3 months.csv",
        6: WHIPLASH_ROOT / "Original_MultiK" / "SD_0.8" / "Whiplash_Events_Original_MultiK_SD_0.8_K6_4-6 months.csv",
        12: WHIPLASH_ROOT / "Original_MultiK" / "SD_0.8" / "Whiplash_Events_Original_MultiK_SD_0.8_K12_7-12 months.csv",
        24: WHIPLASH_ROOT / "Original_MultiK" / "SD_0.8" / "Whiplash_Events_Original_MultiK_SD_0.8_K24_13-24 months.csv",
    },
    "Chronological_First_Opposite": {
        3: WHIPLASH_ROOT / "Chronological_First_Opposite" / "SD_0.8" / "Whiplash_Events_Chronological_First_Opposite_SD_0.8_K3_1-3 months.csv",
        6: WHIPLASH_ROOT / "Chronological_First_Opposite" / "SD_0.8" / "Whiplash_Events_Chronological_First_Opposite_SD_0.8_K6_4-6 months.csv",
        12: WHIPLASH_ROOT / "Chronological_First_Opposite" / "SD_0.8" / "Whiplash_Events_Chronological_First_Opposite_SD_0.8_K12_7-12 months.csv",
        24: WHIPLASH_ROOT / "Chronological_First_Opposite" / "SD_0.8" / "Whiplash_Events_Chronological_First_Opposite_SD_0.8_K24_13-24 months.csv",
    },
    "Shortest_First": {
        3: WHIPLASH_ROOT / "Shortest_First" / "SD_0.8" / "Whiplash_Events_Shortest_First_SD_0.8_K3_1-3 months.csv",
        6: WHIPLASH_ROOT / "Shortest_First" / "SD_0.8" / "Whiplash_Events_Shortest_First_SD_0.8_K6_4-6 months.csv",
        12: WHIPLASH_ROOT / "Shortest_First" / "SD_0.8" / "Whiplash_Events_Shortest_First_SD_0.8_K12_7-12 months.csv",
        24: WHIPLASH_ROOT / "Shortest_First" / "SD_0.8" / "Whiplash_Events_Shortest_First_SD_0.8_K24_13-24 months.csv",
    },
}

GIMMS_PATH = r"D:\Graduate Research\Graduate Research\Data\GIMMS NDVI Data\GIMMS NDVI Data.nc"
SMRZ_DIR = r"D:\Graduate Research\Graduate Research\Data\GLEAM\SMrz_GlEAM_Data"
PPT_DIR = r"D:\Graduate Research\Graduate Research\Data\PRISM_Precipitation_1981_2024"

SOCAL_SHP = r"D:\Graduate Research\Graduate Research\Data\BoundingBox\BoundingBox.shp"

VEG_SHPS = {
    "Agriculture": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Agriculturen_4km.shp",
    "Conifer_Forest": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Conifer_Forest_4km.shp",
    "Conifer_Woodland": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Conifer_Woodland_4km.shp",
    "Desert": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Desert_4km.shp",
    "Hardwood": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Hardwood_4km.shp",
    "Herbaceous": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Herbaceous_4km.shp",
    "Shrub": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Shrub_4km.shp",
    "Urban": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Urban_4km.shp",
}


# ============================================================
# SETTINGS
# ============================================================

DETECTION_METHODS_TO_RUN = ["Original_MultiK"]

K_LIST = [3, 6, 12, 24]

DIRECTIONS = [
    "Dry->Wet",
    "Wet->Dry",
]

VARIABLES_TO_PLOT = [
    "PPT",
    "SMRz",
    "NDVI",
]

HABITAT_ORDER = list(VEG_SHPS.keys())

STUDY_AREA_GROUP = "Study_Area"

GROUP_LABELS = {
    "Agriculture": "Agriculture",
    "Conifer_Forest": "Conifer Forest",
    "Conifer_Woodland": "Conifer Woodland",
    "Desert": "Desert",
    "Hardwood": "Hardwood",
    "Herbaceous": "Herbaceous",
    "Shrub": "Shrub",
    "Urban": "Urban",
    "Study_Area": "Study Area",
}

ANALYSIS_START = "1982-01"
ANALYSIS_END = "2022-12"

ROLLING_WINDOW_MONTHS = 3

COMPOSITE_PRE_START_MONTHS = 3
COMPOSITE_POST_END_MONTHS = 3

ZSCORE_BASELINE_START = "1982-01"
ZSCORE_BASELINE_END = "2022-12"

SAVE_DPI = 400

Y_LIMITS = (-2.5, 2.5)

VARIABLE_COLORS = {
    # Deep Slate Blue for precipitation
    "PPT": "#483D8B",
    # Red for root-zone soil moisture
    "SMRz": "#D7191C",
    # Lighter green for NDVI
    "NDVI": "#66BD63",
}

VARIABLE_LABELS = {
    "PPT": "PPT",
    "SMRz": "SMRz",
    "NDVI": "NDVI",
}

DIRECTION_LABELS = {
    "Dry->Wet": "Dry-to-Wet",
    "Wet->Dry": "Wet-to-Dry",
}


# ============================================================
# ALL-K LANDMARK ANALYSIS SETTINGS
# ============================================================

ALLK_LANDMARK_ANALYSES = [
    {
        "name": "AllK_Landmark_Start1_End1_Previous",
        "folder": FIGURE4_OUT_DIR / "AllK_Landmark_Start1_End1_Previous",
        "labels": [
            "Start -3",
            "Start -2",
            "Start -1",
            "Start",
            "Start +1",
            "End -1",
            "End",
            "End +1",
            "End +2",
            "End +3",
        ],
        "offsets": {
            "Start -3": ("start", -3),
            "Start -2": ("start", -2),
            "Start -1": ("start", -1),
            "Start": ("start", 0),
            "Start +1": ("start", 1),
            "End -1": ("end", -1),
            "End": ("end", 0),
            "End +1": ("end", 1),
            "End +2": ("end", 2),
            "End +3": ("end", 3),
        },
        "priority": [
            "Start -3",
            "Start -2",
            "Start -1",
            "Start",
            "End",
            "End +1",
            "End +2",
            "End +3",
            "Start +1",
            "End -1",
        ],
        "min_transition_duration_months": 0,
        "description": "Previous landmark version: Start +1 and End -1",
        "break_after_label": "Start +1",
        "gap_width": 1.25,
    },
    {
        "name": "AllK_Landmark_Start2_End2_Filtered",
        "folder": FIGURE4_OUT_DIR / "AllK_Landmark_Start2_End2_Filtered",
        "labels": [
            "Start -3",
            "Start -2",
            "Start -1",
            "Start",
            "Start +1",
            "Start +2",
            "End -2",
            "End -1",
            "End",
            "End +1",
            "End +2",
            "End +3",
        ],
        "offsets": {
            "Start -3": ("start", -3),
            "Start -2": ("start", -2),
            "Start -1": ("start", -1),
            "Start": ("start", 0),
            "Start +1": ("start", 1),
            "Start +2": ("start", 2),
            "End -2": ("end", -2),
            "End -1": ("end", -1),
            "End": ("end", 0),
            "End +1": ("end", 1),
            "End +2": ("end", 2),
            "End +3": ("end", 3),
        },
        "priority": [
            "Start -3",
            "Start -2",
            "Start -1",
            "Start",
            "Start +1",
            "Start +2",
            "End -2",
            "End -1",
            "End",
            "End +1",
            "End +2",
            "End +3",
        ],
        # Requires Start +2 and End -2 to be separate interior transition points.
        # This removes K=3 and any too-short K=6 transitions from this analysis.
        "min_transition_duration_months": 5,
        "description": "Expanded middle version: Start +2 and End -2, short transitions removed",
        "break_after_label": "Start +2",
        "gap_width": 1.25,
    },
]

for landmark_cfg in ALLK_LANDMARK_ANALYSES:
    landmark_cfg["fig_dir"] = landmark_cfg["folder"] / "Figures"
    landmark_cfg["csv_dir"] = landmark_cfg["folder"] / "CSV"
    landmark_cfg["fig_dir"].mkdir(parents=True, exist_ok=True)
    landmark_cfg["csv_dir"].mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_name(text):
    text = str(text)
    text = text.replace("->", "To")
    text = text.replace("|", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def group_display_name(group):
    return GROUP_LABELS.get(group, str(group).replace("_", " "))


def month_range(start_ym, end_ym):
    return pd.period_range(
        pd.Period(start_ym, freq="M"),
        pd.Period(end_ym, freq="M"),
        freq="M"
    ).strftime("%Y-%m").tolist()


def period_month_diff(later_ym, earlier_ym):
    later = pd.Period(later_ym, freq="M")
    earlier = pd.Period(earlier_ym, freq="M")

    return int((later.year - earlier.year) * 12 + (later.month - earlier.month))


def clean_geometries(gdf):
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    return gdf


def get_datetime_index(da):
    idx = da.indexes["time"]

    if hasattr(idx, "to_datetimeindex"):
        return idx.to_datetimeindex()

    return pd.to_datetime(idx)


def standardize_monthly_time(da):
    if "time" not in da.dims:
        return da

    time_vals = get_datetime_index(da)
    month_start = pd.PeriodIndex(time_vals, freq="M").to_timestamp()

    da = da.assign_coords(time=("time", month_start))

    if pd.Index(month_start).duplicated().any():
        da = da.groupby("time").mean("time", skipna=True)

    da = da.sortby("time")

    time_vals2 = pd.DatetimeIndex(pd.to_datetime(da["time"].values))

    da = da.assign_coords(
        ym=("time", time_vals2.strftime("%Y-%m"))
    )

    return da


def normalize_longitude_if_needed(da):
    lon_name = None

    for candidate in ["x", "lon", "longitude"]:
        if candidate in da.coords or candidate in da.dims:
            lon_name = candidate
            break

    if lon_name is not None:
        vals = da[lon_name].values

        if np.nanmax(vals) > 180:
            da = da.assign_coords({
                lon_name: (((da[lon_name] + 180) % 360) - 180)
            })
            da = da.sortby(lon_name)

    return da


def rename_spatial_dims_to_xy(da):
    rename_dict = {}

    if "lon" in da.dims:
        rename_dict["lon"] = "x"

    if "longitude" in da.dims:
        rename_dict["longitude"] = "x"

    if "lat" in da.dims:
        rename_dict["lat"] = "y"

    if "latitude" in da.dims:
        rename_dict["latitude"] = "y"

    if len(rename_dict) > 0:
        da = da.rename(rename_dict)

    return da


def ensure_spatial_dims(da):
    possible_x = ["x", "lon", "longitude"]
    possible_y = ["y", "lat", "latitude"]

    x_dim = None
    y_dim = None

    for dim in da.dims:
        if dim.lower() in possible_x:
            x_dim = dim

        if dim.lower() in possible_y:
            y_dim = dim

    if x_dim is not None and y_dim is not None:
        da = da.rio.set_spatial_dims(
            x_dim=x_dim,
            y_dim=y_dim,
            inplace=False
        )

    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da


def subset_by_ym(da, months_keep):
    da = standardize_monthly_time(da)
    mask = np.isin(da["ym"].values, months_keep)

    return da.isel(time=mask)


def get_composite_load_months():
    start_period = (
        pd.Period(ANALYSIS_START, freq="M")
        - COMPOSITE_PRE_START_MONTHS
        - (ROLLING_WINDOW_MONTHS - 1)
    )

    end_period = (
        pd.Period(ANALYSIS_END, freq="M")
        + max(K_LIST)
        + COMPOSITE_POST_END_MONTHS
    )

    return pd.period_range(start_period, end_period, freq="M").strftime("%Y-%m").tolist()


def reindex_to_months(cube, months_keep):
    cube = standardize_monthly_time(cube)

    full_time = pd.PeriodIndex(months_keep, freq="M").to_timestamp()

    cube = cube.reindex(time=full_time)

    cube = cube.assign_coords(
        ym=("time", pd.DatetimeIndex(full_time).strftime("%Y-%m"))
    )

    return cube


def apply_rolling_and_monthly_zscore(cube, months_keep):
    cube = standardize_monthly_time(cube)
    cube = reindex_to_months(cube, months_keep)

    if ROLLING_WINDOW_MONTHS is not None and ROLLING_WINDOW_MONTHS > 1:
        cube = cube.rolling(
            time=ROLLING_WINDOW_MONTHS,
            min_periods=ROLLING_WINDOW_MONTHS
        ).mean()

    cube = standardize_monthly_time(cube)

    baseline_months = month_range(ZSCORE_BASELINE_START, ZSCORE_BASELINE_END)
    baseline = subset_by_ym(cube, baseline_months)

    clim_mean = baseline.groupby("time.month").mean("time", skipna=True)
    clim_std = baseline.groupby("time.month").std("time", skipna=True)
    clim_std = clim_std.where(clim_std > 0)

    z = cube.groupby("time.month") - clim_mean
    z = z.groupby("time.month") / clim_std

    z = standardize_monthly_time(z)
    z = z.astype("float32").load()

    return z


def parse_ym_from_bil(filepath):
    fname = os.path.basename(filepath)

    m = re.search(r"_(\d{6})_", fname)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}-{raw[4:]}"

    m = re.search(r"(\d{6})", fname)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}-{raw[4:]}"

    return None


def print_cube_diagnostics(cube, name):
    cube = standardize_monthly_time(cube)

    ym_vals = cube["ym"].values

    finite_count = int(np.isfinite(cube.values).sum())
    total_count = int(np.prod(cube.shape))

    print(f"\n{name} diagnostics:")
    print(f"  shape: {cube.shape}")

    if len(ym_vals) > 0:
        print(f"  time range: {ym_vals[0]} to {ym_vals[-1]}")

    print(f"  n months: {len(ym_vals)}")
    print(f"  finite values: {finite_count} / {total_count}")

    return cube


# ============================================================
# STATIC HABITAT / STUDY-AREA LAYERS
# ============================================================

def create_pixel_area_grid_km2(template_da):
    template_da = ensure_spatial_dims(template_da)

    crs = template_da.rio.crs
    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape

    if crs is not None and crs.is_projected:
        pixel_area = abs(transform.a * transform.e) / 1_000_000.0
        return np.full((height, width), pixel_area, dtype=float)

    geod = Geod(ellps="WGS84")
    areas = np.full((height, width), np.nan, dtype=float)

    for r in range(height):
        for c in range(width):
            x_ul, y_ul = xy(transform, r, c, offset="ul")
            x_ur, y_ur = xy(transform, r, c, offset="ur")
            x_lr, y_lr = xy(transform, r, c, offset="lr")
            x_ll, y_ll = xy(transform, r, c, offset="ll")

            area_m2, _ = geod.polygon_area_perimeter(
                [x_ul, x_ur, x_lr, x_ll],
                [y_ul, y_ur, y_lr, y_ll]
            )

            areas[r, c] = abs(area_m2) / 1_000_000.0

    return areas


def make_pixel_center_table(template_da):
    template_da = ensure_spatial_dims(template_da)

    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape

    records = []

    for row in range(height):
        for col in range(width):
            lon, lat = xy(transform, row, col, offset="center")

            records.append({
                "row": int(row),
                "col": int(col),
                "lon": float(lon),
                "lat": float(lat),
            })

    df = pd.DataFrame.from_records(records)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=template_da.rio.crs
    )

    return gdf


def load_habitat_polygons():
    pieces = []

    for habitat, shp in VEG_SHPS.items():
        shp_path = Path(shp)

        if not shp_path.exists():
            warnings.warn(f"Missing habitat shapefile: {shp_path}")
            continue

        gdf = gpd.read_file(shp_path)

        if gdf.crs is None:
            raise ValueError(f"Habitat shapefile has no CRS: {shp_path}")

        gdf = clean_geometries(gdf)
        gdf["habitat"] = habitat

        pieces.append(gdf[["habitat", "geometry"]])

    if len(pieces) == 0:
        raise ValueError("No habitat shapefiles were loaded.")

    habitats = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True),
        crs=pieces[0].crs
    )

    habitats = clean_geometries(habitats)

    return habitats


def prepare_static_habitat_layers(template_da):
    print("Preparing habitat and study-area layer using pixel-center joins...")

    template_da = ensure_spatial_dims(template_da)

    pixel_points = make_pixel_center_table(template_da)
    habitat_gdf = load_habitat_polygons().to_crs(pixel_points.crs)

    joined = gpd.sjoin(
        pixel_points,
        habitat_gdf,
        how="left",
        predicate="within"
    )

    joined = (
        joined
        .sort_values(["row", "col"])
        .drop_duplicates(["row", "col"])
    )

    static = pixel_points[["row", "col", "lon", "lat", "geometry"]].copy()

    static = static.merge(
        joined[["row", "col", "habitat"]],
        on=["row", "col"],
        how="left"
    )

    area_grid = create_pixel_area_grid_km2(template_da)

    area_records = []

    for row in range(area_grid.shape[0]):
        for col in range(area_grid.shape[1]):
            area_records.append({
                "row": int(row),
                "col": int(col),
                "area_km2": float(area_grid[row, col])
            })

    area_df = pd.DataFrame(area_records)

    static = static.merge(area_df, on=["row", "col"], how="left")

    habitat_order_present = [
        h for h in HABITAT_ORDER
        if h in static["habitat"].dropna().unique()
    ]

    plot_group_order = habitat_order_present + [STUDY_AREA_GROUP]

    print("\nHabitat counts assigned to grid pixel centers:")
    print(static["habitat"].value_counts(dropna=False))

    print("\nPlot groups:")
    for g in plot_group_order:
        print(f"  {g}")

    return {
        "static_pixel_table": static.drop(columns="geometry"),
        "habitat_order": habitat_order_present,
        "plot_group_order": plot_group_order,
    }


# ============================================================
# DATA LOADERS
# ============================================================

def load_ndvi_base(gimms_path, clip_gdf):
    print("Loading NDVI base...")

    if os.path.isdir(gimms_path):
        files = sorted(glob.glob(os.path.join(gimms_path, "*.nc*")))

        if len(files) == 0:
            raise FileNotFoundError(
                f"No .nc files found in folder: {gimms_path}")

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override"
        )

    else:
        if not os.path.exists(gimms_path):
            raise FileNotFoundError(f"NDVI path not found: {gimms_path}")

        ds = xr.open_dataset(gimms_path)

    ndvi_var = [v for v in ds.data_vars if "NDVI" in v.upper()]

    if len(ndvi_var) == 0:
        raise ValueError("No NDVI variable found in NDVI dataset.")

    ndvi = ds[ndvi_var[0]]

    ndvi = rename_spatial_dims_to_xy(ndvi)
    ndvi = normalize_longitude_if_needed(ndvi)
    ndvi = ensure_spatial_dims(ndvi)

    ndvi = ndvi.where(ndvi > -2000) * 0.0001

    ndvi = ndvi.rio.clip(
        clip_gdf.geometry,
        clip_gdf.crs,
        drop=True
    )

    ndvi = standardize_monthly_time(ndvi)
    ndvi = ndvi.astype("float32")

    print_cube_diagnostics(ndvi, "NDVI base after monthly standardization")

    return ndvi


def load_ndvi_composite_cube(ndvi_base, months_load):
    print("\nPreparing NDVI composite cube...")

    ndvi = subset_by_ym(ndvi_base, months_load)
    ndvi = apply_rolling_and_monthly_zscore(ndvi, months_load)

    print_cube_diagnostics(ndvi, "NDVI 3-month rolling z-score")

    return ndvi


def load_prism_composite_cube(folder, template_2d, months_load, var_name="PRISM"):
    print(f"\nLoading {var_name} composite cube...")

    rasters = []
    months = []
    times = []

    files = sorted(glob.glob(os.path.join(folder, "*.bil")))

    for f in files:
        ym = parse_ym_from_bil(f)

        if ym is None or ym not in months_load:
            continue

        da = rxr.open_rasterio(f, masked=True).squeeze()
        da = da.rio.reproject_match(template_2d)
        da = da.astype("float32")

        rasters.append(da)
        months.append(ym)
        times.append(pd.Period(ym, freq="M").to_timestamp())

    if len(rasters) == 0:
        raise ValueError(f"No valid raster slices loaded from {folder}")

    order = np.argsort(times)

    rasters = [rasters[i] for i in order]
    months = [months[i] for i in order]
    times = [times[i] for i in order]

    cube = xr.concat(rasters, dim="time")

    cube = cube.assign_coords(
        time=pd.to_datetime(times),
        ym=("time", months)
    )

    cube = apply_rolling_and_monthly_zscore(cube, months_load)

    print_cube_diagnostics(cube, f"{var_name} 3-month rolling z-score")

    return cube


def load_smrz_composite_cube(smrz_dir, template_2d, clip_gdf, months_load):
    print("\nLoading SMRz composite cube, memory-safe...")

    smrz_files = sorted(glob.glob(os.path.join(smrz_dir, "*.nc*")))

    if len(smrz_files) == 0:
        raise FileNotFoundError(
            f"No .nc files found in SMRz folder: {smrz_dir}")

    ds = xr.open_mfdataset(
        smrz_files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override"
    )

    smrz_var = [v for v in ds.data_vars if "SMR" in v.upper()]

    if len(smrz_var) == 0:
        smrz_var = [
            v for v in ds.data_vars
            if "SOIL" in v.upper()
            or "MOIST" in v.upper()
            or "ROOT" in v.upper()
        ]

    if len(smrz_var) == 0:
        raise ValueError("No SMRz variable found in dataset.")

    smrz = ds[smrz_var[0]]

    allowed_dims = ["time", "x", "y", "lon", "lat", "longitude", "latitude"]

    for dim in list(smrz.dims):
        if dim not in allowed_dims:
            print(f"Selecting first index for extra SMRz dimension: {dim}")
            smrz = smrz.isel({dim: 0})

    smrz = rename_spatial_dims_to_xy(smrz)
    smrz = normalize_longitude_if_needed(smrz)
    smrz = ensure_spatial_dims(smrz)
    smrz = standardize_monthly_time(smrz)

    out_slices = []
    out_months = []
    out_times = []

    ym_all = smrz["ym"].values

    for t_idx, ym in enumerate(ym_all):
        if ym not in months_load:
            continue

        da_t = smrz.isel(time=t_idx)

        da_t = da_t.rio.clip(
            clip_gdf.geometry,
            clip_gdf.crs,
            drop=True
        )

        da_t = da_t.rio.reproject_match(template_2d)
        da_t = da_t.astype("float32")

        out_slices.append(da_t)
        out_months.append(ym)
        out_times.append(pd.Period(str(ym), freq="M").to_timestamp())

    if len(out_slices) == 0:
        raise ValueError("No SMRz slices matched requested months.")

    cube = xr.concat(out_slices, dim="time")

    cube = cube.assign_coords(
        time=pd.to_datetime(out_times),
        ym=("time", out_months)
    )

    cube = apply_rolling_and_monthly_zscore(cube, months_load)

    print_cube_diagnostics(cube, "SMRz 3-month rolling z-score")

    return cube


# ============================================================
# GROUP-WEIGHTED TIME SERIES
# ============================================================

def get_group_weighted_series(cube, static_df, group_name):
    if group_name == STUDY_AREA_GROUP:
        df = static_df.loc[
            np.isfinite(static_df["area_km2"])
            & (static_df["area_km2"] > 0)
        ].copy()
    else:
        df = static_df.loc[
            (static_df["habitat"] == group_name)
            & np.isfinite(static_df["area_km2"])
            & (static_df["area_km2"] > 0)
        ].copy()

    if df.empty:
        return pd.Series(dtype=float)

    rows = df["row"].astype(int).values
    cols = df["col"].astype(int).values
    weights = df["area_km2"].astype(float).values

    values = cube.values[:, rows, cols]

    finite = np.isfinite(values)

    weighted_values = np.where(
        finite,
        values * weights[None, :],
        0.0
    )

    denom = np.where(
        finite,
        weights[None, :],
        0.0
    ).sum(axis=1)

    numerator = weighted_values.sum(axis=1)

    mean_values = np.full(values.shape[0], np.nan, dtype=float)

    good = denom > 0
    mean_values[good] = numerator[good] / denom[good]

    return pd.Series(mean_values, index=cube["ym"].values)


def build_group_timeseries_table(cube_by_variable, static_layers):
    print("\nBuilding group-weighted 3-month rolling z-score time series...")

    static_df = static_layers["static_pixel_table"]
    plot_group_order = static_layers["plot_group_order"]

    records = []

    for variable, cube in cube_by_variable.items():
        print(f"  Group time series for {variable}")

        for group_name in plot_group_order:
            series = get_group_weighted_series(
                cube=cube,
                static_df=static_df,
                group_name=group_name
            )

            if series.empty:
                continue

            for ym, value in series.items():
                records.append({
                    "group": group_name,
                    "group_label": group_display_name(group_name),
                    "variable": variable,
                    "ym": str(ym),
                    "value_z": float(value) if np.isfinite(value) else np.nan,
                })

    ts_df = pd.DataFrame.from_records(records)

    out_csv = FIGURE4_CSV_DIR / (
        f"Group_3MonthRolling_Zscore_TimeSeries_"
        f"{safe_name(ANALYSIS_START)}_{safe_name(ANALYSIS_END)}.csv"
    )

    ts_df.to_csv(out_csv, index=False)
    print(f"Saved group time series CSV: {out_csv}")

    return ts_df


# ============================================================
# EVENT HELPERS
# ============================================================

def normalize_direction(value):
    s = str(value).lower()
    s = s.replace(" ", "")
    s = s.replace("-", "")
    s = s.replace("_", "")
    s = s.replace(">", "to")

    if "dry" in s and "wet" in s:
        if s.find("dry") < s.find("wet"):
            return "Dry->Wet"

        return "Wet->Dry"

    return str(value)


def load_events_for_method_k(detection_method, k_value):
    event_path = WHIPLASH_EVENT_FILES[detection_method][k_value]

    if not event_path.exists():
        raise FileNotFoundError(f"Missing event file:\n{event_path}")

    event_df = pd.read_csv(event_path)

    required = {"StartDate", "EndDate", "Direction"}
    missing = sorted(required - set(event_df.columns))

    if missing:
        raise ValueError(
            f"Whiplash event file is missing required columns {missing}:\n{event_path}"
        )

    event_df = event_df.copy()
    event_df["StartDate"] = pd.to_datetime(event_df["StartDate"])
    event_df["EndDate"] = pd.to_datetime(event_df["EndDate"])

    event_df["direction_norm"] = event_df["Direction"].apply(
        normalize_direction)
    event_df["K"] = int(k_value)
    event_df["detection_method"] = detection_method

    event_df["start_ym"] = event_df["StartDate"].dt.to_period("M").astype(str)
    event_df["end_ym"] = event_df["EndDate"].dt.to_period("M").astype(str)

    event_df = event_df.loc[
        (event_df["start_ym"] >= ANALYSIS_START)
        & (event_df["end_ym"] <= ANALYSIS_END)
    ].copy()

    event_df["transition_duration_months"] = event_df.apply(
        lambda row: period_month_diff(row["end_ym"], row["start_ym"]),
        axis=1
    )

    event_df = event_df.reset_index(drop=True)
    event_df["event_id"] = np.arange(1, len(event_df) + 1)

    return event_df


def load_all_events():
    pieces = []

    for detection_method in DETECTION_METHODS_TO_RUN:
        for k_value in K_LIST:
            event_df = load_events_for_method_k(detection_method, k_value)
            pieces.append(event_df)

    if len(pieces) == 0:
        raise ValueError("No event files were loaded.")

    all_events = pd.concat(pieces, ignore_index=True)

    all_events["global_event_id"] = np.arange(1, len(all_events) + 1)

    return all_events


def make_series_lookup(ts_df):
    lookup = {}

    for (group_name, variable), sub in ts_df.groupby(["group", "variable"]):
        s = pd.Series(
            sub["value_z"].values,
            index=sub["ym"].astype(str).values
        )

        lookup[(group_name, variable)] = s

    return lookup


# ============================================================
# K-SPECIFIC REAL-MONTH COMPOSITES
# ============================================================

def build_real_month_composite_event_values(ts_df, plot_group_order):
    print("\nBuilding K-specific real-month composite table...")

    series_lookup = make_series_lookup(ts_df)

    records = []

    all_events = load_all_events()

    for detection_method in DETECTION_METHODS_TO_RUN:
        for k_value in K_LIST:
            event_df = all_events.loc[
                (all_events["detection_method"] == detection_method)
                & (all_events["K"] == k_value)
            ].copy()

            print("\n" + "=" * 70)
            print(
                f"Real-month composite events: {detection_method}, K={k_value}")
            print("=" * 70)

            for direction in DIRECTIONS:
                df_dir = event_df.loc[
                    event_df["direction_norm"] == direction
                ].copy()

                print(
                    f"  {DIRECTION_LABELS.get(direction, direction)}: "
                    f"{len(df_dir)} events"
                )

                if df_dir.empty:
                    continue

                for _, event in df_dir.iterrows():
                    start_period = pd.Period(event["start_ym"], freq="M")
                    end_period = pd.Period(event["end_ym"], freq="M")

                    start_rel = -COMPOSITE_PRE_START_MONTHS
                    end_rel = (
                        int(event["transition_duration_months"])
                        + COMPOSITE_POST_END_MONTHS
                    )

                    for rel_month in range(start_rel, end_rel + 1):
                        current_period = start_period + rel_month
                        ym = current_period.strftime("%Y-%m")

                        in_transition = (
                            current_period >= start_period
                            and current_period <= end_period
                        )

                        for group_name in plot_group_order:
                            for variable in VARIABLES_TO_PLOT:
                                key = (group_name, variable)

                                if key not in series_lookup:
                                    value = np.nan
                                else:
                                    series = series_lookup[key]
                                    value = series.loc[ym] if ym in series.index else np.nan

                                records.append({
                                    "composite_type": "K_specific_real_month",
                                    "detection_method": detection_method,
                                    "K": int(k_value),
                                    "direction": direction,
                                    "direction_label": DIRECTION_LABELS.get(direction, direction),
                                    "event_id": int(event["event_id"]),
                                    "global_event_id": int(event["global_event_id"]),
                                    "event_start": event["StartDate"],
                                    "event_end": event["EndDate"],
                                    "start_ym": event["start_ym"],
                                    "end_ym": event["end_ym"],
                                    "transition_duration_months": int(event["transition_duration_months"]),
                                    "relative_month_from_start": int(rel_month),
                                    "calendar_ym": ym,
                                    "in_transition_window": bool(in_transition),
                                    "group": group_name,
                                    "group_label": group_display_name(group_name),
                                    "variable": variable,
                                    "value_z": float(value) if np.isfinite(value) else np.nan,
                                })

    event_values = pd.DataFrame.from_records(records)

    out_csv = FIGURE4_CSV_DIR / (
        f"Group_Composite_EventValues_KSpecific_RealMonth_"
        f"StartMinus{COMPOSITE_PRE_START_MONTHS}_"
        f"EndPlus{COMPOSITE_POST_END_MONTHS}_"
        f"3MonthRollingZ_{safe_name(ANALYSIS_START)}_{safe_name(ANALYSIS_END)}.csv"
    )

    event_values.to_csv(out_csv, index=False)
    print(f"\nSaved K-specific real-month event-value CSV: {out_csv}")

    return event_values


def summarize_real_month_composites(event_values):
    print("\nSummarizing K-specific real-month composites...")

    group_cols = [
        "detection_method",
        "K",
        "direction",
        "direction_label",
        "relative_month_from_start",
        "group",
        "group_label",
        "variable",
    ]

    summary = (
        event_values
        .groupby(group_cols, dropna=False)
        .agg(
            mean_z=("value_z", "mean"),
            median_z=("value_z", "median"),
            p25_z=("value_z", lambda x: np.nanpercentile(
                x, 25) if np.isfinite(x).any() else np.nan),
            p75_z=("value_z", lambda x: np.nanpercentile(
                x, 75) if np.isfinite(x).any() else np.nan),
            std_z=("value_z", "std"),
            n_events_with_data=(
                "value_z", lambda x: int(np.isfinite(x).sum())),
        )
        .reset_index()
    )

    out_csv = FIGURE4_CSV_DIR / (
        f"Group_Composite_Summary_KSpecific_RealMonth_"
        f"StartMinus{COMPOSITE_PRE_START_MONTHS}_"
        f"EndPlus{COMPOSITE_POST_END_MONTHS}_"
        f"3MonthRollingZ_{safe_name(ANALYSIS_START)}_{safe_name(ANALYSIS_END)}.csv"
    )

    summary.to_csv(out_csv, index=False)
    print(f"Saved K-specific real-month summary CSV: {out_csv}")

    return summary


# ============================================================
# ALL-K LANDMARK COMPOSITES
# ============================================================

def get_landmark_order_map(landmark_cfg):
    return {
        label: i
        for i, label in enumerate(landmark_cfg["labels"])
    }


def get_landmark_x_position_map(landmark_cfg):
    labels = landmark_cfg["labels"]
    break_after_label = landmark_cfg.get("break_after_label", None)
    gap_width = float(landmark_cfg.get("gap_width", 0.0))

    x_map = {}
    shift = 0.0

    for i, label in enumerate(labels):
        x_map[label] = float(i) + shift

        if label == break_after_label:
            shift += gap_width

    return x_map


def get_landmark_periods_for_event(event, landmark_cfg):
    start_period = pd.Period(event["start_ym"], freq="M")
    end_period = pd.Period(event["end_ym"], freq="M")

    order_map = get_landmark_order_map(landmark_cfg)

    candidates = {}

    for label, offset_info in landmark_cfg["offsets"].items():
        anchor_name, offset = offset_info

        if anchor_name == "start":
            period = start_period + offset
        elif anchor_name == "end":
            period = end_period + offset
        else:
            raise ValueError(f"Unknown landmark anchor: {anchor_name}")

        candidates[label] = {
            "period": period,
            "segment": anchor_name,
        }

    used_months = set()
    kept = []

    for label in landmark_cfg["priority"]:
        if label not in candidates:
            continue

        period = candidates[label]["period"]
        segment = candidates[label]["segment"]
        ym = period.strftime("%Y-%m")

        if ym in used_months:
            continue

        used_months.add(ym)

        kept.append({
            "landmark_label": label,
            "landmark_order": order_map[label],
            "landmark_segment": segment,
            "calendar_ym": ym,
            "calendar_period": period,
        })

    kept = sorted(kept, key=lambda d: d["landmark_order"])

    return kept


def build_landmark_composite_event_values(ts_df, plot_group_order, landmark_cfg):
    print("\nBuilding All-K landmark composite table...")
    print(f"  Analysis: {landmark_cfg['name']}")
    print(f"  Description: {landmark_cfg['description']}")
    print(f"  Output folder: {landmark_cfg['folder']}")

    series_lookup = make_series_lookup(ts_df)

    all_events = load_all_events()

    min_duration = landmark_cfg.get("min_transition_duration_months", 0)

    if min_duration > 0:
        before_count = len(all_events)

        all_events = all_events.loc[
            all_events["transition_duration_months"] >= min_duration
        ].copy()

        after_count = len(all_events)

        print(
            f"  Filtered events for expanded landmark analysis: "
            f"kept {after_count} of {before_count} transition records "
            f"(minimum duration = {min_duration} months)"
        )

        if after_count > 0:
            print("  Kept events by K:")
            print(all_events["K"].value_counts().sort_index())

    records = []

    for detection_method in DETECTION_METHODS_TO_RUN:
        method_events = all_events.loc[
            all_events["detection_method"] == detection_method
        ].copy()

        print("\n" + "=" * 70)
        print(f"All-K landmark composite events: {detection_method}")
        print(f"Analysis: {landmark_cfg['name']}")
        print("=" * 70)

        for direction in DIRECTIONS:
            df_dir = method_events.loc[
                method_events["direction_norm"] == direction
            ].copy()

            print(
                f"  {DIRECTION_LABELS.get(direction, direction)}: "
                f"{len(df_dir)} events across eligible K values"
            )

            if df_dir.empty:
                continue

            for _, event in df_dir.iterrows():
                landmarks = get_landmark_periods_for_event(
                    event=event,
                    landmark_cfg=landmark_cfg
                )

                for landmark in landmarks:
                    ym = landmark["calendar_ym"]

                    for group_name in plot_group_order:
                        for variable in VARIABLES_TO_PLOT:
                            key = (group_name, variable)

                            if key not in series_lookup:
                                value = np.nan
                            else:
                                series = series_lookup[key]
                                value = series.loc[ym] if ym in series.index else np.nan

                            records.append({
                                "composite_type": landmark_cfg["name"],
                                "landmark_description": landmark_cfg["description"],
                                "detection_method": detection_method,
                                "K": int(event["K"]),
                                "direction": direction,
                                "direction_label": DIRECTION_LABELS.get(direction, direction),
                                "event_id": int(event["event_id"]),
                                "global_event_id": int(event["global_event_id"]),
                                "event_start": event["StartDate"],
                                "event_end": event["EndDate"],
                                "start_ym": event["start_ym"],
                                "end_ym": event["end_ym"],
                                "transition_duration_months": int(event["transition_duration_months"]),
                                "landmark_label": landmark["landmark_label"],
                                "landmark_order": int(landmark["landmark_order"]),
                                "landmark_segment": landmark["landmark_segment"],
                                "calendar_ym": ym,
                                "group": group_name,
                                "group_label": group_display_name(group_name),
                                "variable": variable,
                                "value_z": float(value) if np.isfinite(value) else np.nan,
                            })

    event_values = pd.DataFrame.from_records(records)

    out_csv = landmark_cfg["csv_dir"] / (
        f"Group_Composite_EventValues_{safe_name(landmark_cfg['name'])}_"
        f"3MonthRollingZ_{safe_name(ANALYSIS_START)}_{safe_name(ANALYSIS_END)}.csv"
    )

    event_values.to_csv(out_csv, index=False)
    print(f"\nSaved All-K landmark event-value CSV: {out_csv}")

    return event_values


def summarize_landmark_composites(event_values, landmark_cfg):
    print("\nSummarizing All-K landmark composites...")
    print(f"  Analysis: {landmark_cfg['name']}")

    if event_values.empty:
        print("  No event values available for this landmark analysis.")
        return pd.DataFrame()

    group_cols = [
        "detection_method",
        "direction",
        "direction_label",
        "landmark_order",
        "landmark_label",
        "landmark_segment",
        "group",
        "group_label",
        "variable",
    ]

    summary = (
        event_values
        .groupby(group_cols, dropna=False)
        .agg(
            mean_z=("value_z", "mean"),
            median_z=("value_z", "median"),
            p25_z=("value_z", lambda x: np.nanpercentile(
                x, 25) if np.isfinite(x).any() else np.nan),
            p75_z=("value_z", lambda x: np.nanpercentile(
                x, 75) if np.isfinite(x).any() else np.nan),
            std_z=("value_z", "std"),
            n_events_with_data=(
                "value_z", lambda x: int(np.isfinite(x).sum())),
        )
        .reset_index()
    )

    out_csv = landmark_cfg["csv_dir"] / (
        f"Group_Composite_Summary_{safe_name(landmark_cfg['name'])}_"
        f"3MonthRollingZ_{safe_name(ANALYSIS_START)}_{safe_name(ANALYSIS_END)}.csv"
    )

    summary.to_csv(out_csv, index=False)
    print(f"Saved All-K landmark summary CSV: {out_csv}")

    return summary


# ============================================================
# PLOTTING HELPERS
# ============================================================

def get_y_limits_for_plot(summary_sub):
    if Y_LIMITS is not None:
        return Y_LIMITS

    vals = []

    for col in ["p25_z", "p75_z", "median_z"]:
        if col in summary_sub.columns:
            vals.extend(summary_sub[col].values.tolist())

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return (-2.5, 2.5)

    ymin = np.nanpercentile(vals, 2)
    ymax = np.nanpercentile(vals, 98)

    pad = 0.25 * max(abs(ymin), abs(ymax), 1.0)

    return (ymin - pad, ymax + pad)


def choose_xticks(xmin, xmax):
    """
    Choose labeled x-axis ticks.

    For short K-specific plots, label every month so intermediate
    months such as -2, -1, 1, 2, 4, and 5 are visible. For longer
    transition windows, use a wider label step to avoid overcrowding.
    """
    span = xmax - xmin

    if span <= 12:
        step = 1
    elif span <= 18:
        step = 2
    elif span <= 36:
        step = 3
    else:
        step = 6

    start = int(np.floor(xmin / step) * step)
    end = int(np.ceil(xmax / step) * step)

    return np.arange(start, end + 1, step)


def choose_minor_xticks(xmin, xmax):
    """
    Add minor tick marks at every month even when not every month is labeled.
    """
    start = int(np.floor(xmin))
    end = int(np.ceil(xmax))

    return np.arange(start, end + 1, 1)


def get_real_month_event_metadata(event_values, detection_method, k_value, direction):
    sub = event_values.loc[
        (event_values["detection_method"] == detection_method)
        & (event_values["K"] == k_value)
        & (event_values["direction"] == direction)
    ].copy()

    if sub.empty:
        return {
            "n_events": 0,
            "median_duration": np.nan,
            "p25_duration": np.nan,
            "p75_duration": np.nan,
            "max_relative_month": COMPOSITE_PRE_START_MONTHS + COMPOSITE_POST_END_MONTHS,
        }

    event_meta = (
        sub[["global_event_id", "transition_duration_months"]]
        .drop_duplicates()
        .copy()
    )

    durations = event_meta["transition_duration_months"].astype(float).values

    return {
        "n_events": int(event_meta["global_event_id"].nunique()),
        "median_duration": float(np.nanmedian(durations)),
        "p25_duration": float(np.nanpercentile(durations, 25)),
        "p75_duration": float(np.nanpercentile(durations, 75)),
        "max_relative_month": int(sub["relative_month_from_start"].max()),
    }


def get_landmark_event_count(event_values, detection_method, direction):
    sub = event_values.loc[
        (event_values["detection_method"] == detection_method)
        & (event_values["direction"] == direction)
    ].copy()

    if sub.empty:
        return 0

    return int(sub["global_event_id"].nunique())


# ============================================================
# K-SPECIFIC REAL-MONTH PLOTS
# ============================================================

def plot_one_real_month_grid(summary, event_values, detection_method, k_value, direction, plot_group_order):
    summary_sub = summary.loc[
        (summary["detection_method"] == detection_method)
        & (summary["K"] == k_value)
        & (summary["direction"] == direction)
    ].copy()

    if summary_sub.empty:
        print(
            f"No real-month summary data for {detection_method}, K={k_value}, {direction}. Skipping plot.")
        return

    meta = get_real_month_event_metadata(
        event_values=event_values,
        detection_method=detection_method,
        k_value=k_value,
        direction=direction
    )

    n_events = meta["n_events"]
    median_duration = meta["median_duration"]
    p25_duration = meta["p25_duration"]
    p75_duration = meta["p75_duration"]
    max_relative_month = meta["max_relative_month"]

    ncols = 3
    nrows = int(np.ceil(len(plot_group_order) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    y_min, y_max = get_y_limits_for_plot(summary_sub)

    x_min = -COMPOSITE_PRE_START_MONTHS
    x_max = max_relative_month
    xticks = choose_xticks(x_min, x_max)
    minor_xticks = choose_minor_xticks(x_min, x_max)

    handles = []
    labels = []

    for ax, group_name in zip(axes, plot_group_order):
        group_sub = summary_sub.loc[
            summary_sub["group"] == group_name
        ].copy()

        if np.isfinite(median_duration):
            ax.axvspan(
                0,
                median_duration,
                color="#bdbdbd",
                alpha=0.18,
                linewidth=0,
                zorder=0
            )

        ax.axvline(
            0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            zorder=2
        )

        if np.isfinite(median_duration):
            ax.axvline(
                median_duration,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                zorder=2
            )

        for variable in VARIABLES_TO_PLOT:
            var_sub = group_sub.loc[
                group_sub["variable"] == variable
            ].copy()

            if var_sub.empty:
                continue

            var_sub = var_sub.sort_values("relative_month_from_start")

            x = var_sub["relative_month_from_start"].values.astype(float)
            center = var_sub["median_z"].values.astype(float)
            p25 = var_sub["p25_z"].values.astype(float)
            p75 = var_sub["p75_z"].values.astype(float)

            color = VARIABLE_COLORS.get(variable, None)
            label = VARIABLE_LABELS.get(variable, variable)

            ax.fill_between(
                x,
                p25,
                p75,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1
            )

            line, = ax.plot(
                x,
                center,
                color=color,
                linewidth=1.8,
                label=label,
                zorder=3
            )

            if label not in labels:
                handles.append(line)
                labels.append(label)

        ax.axhline(
            0,
            color="black",
            linewidth=0.7,
            linestyle="-",
            alpha=0.65,
            zorder=2
        )

        ax.set_title(group_display_name(group_name), fontsize=11)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(xticks)
        ax.set_xticks(minor_xticks, minor=True)
        ax.tick_params(axis="x", which="minor", length=3, width=0.6)

        ax.grid(True, alpha=0.25, linewidth=0.7)

    for ax in axes[len(plot_group_order):]:
        ax.set_axis_off()

    for ax in axes:
        ax.tick_params(labelsize=8)

    for ax in axes[-3:]:
        ax.set_xlabel("Months since whiplash StartDate", fontsize=9)

    for row_i in range(nrows):
        axes[row_i * 3].set_ylabel("3-month rolling z-score", fontsize=9)

    direction_label = DIRECTION_LABELS.get(direction, direction)

    if np.isfinite(median_duration):
        duration_text = (
            f"median transition duration = {median_duration:.1f} months "
            f"(IQR {p25_duration:.1f}-{p75_duration:.1f})"
        )
    else:
        duration_text = "transition duration unavailable"

    fig.suptitle(
        f"K-specific habitat and study-area composite around {direction_label} whiplash transitions\n"
        f"{detection_method}, K={k_value}, StartDate - {COMPOSITE_PRE_START_MONTHS} months "
        f"to EndDate + {COMPOSITE_POST_END_MONTHS} months, n={n_events} events\n"
        f"Line = median; shading = 25th-75th percentile; gray band = typical transition period; "
        f"{duration_text}",
        fontsize=13,
        y=0.995
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.90])

    out_png = FIGURE4_FIG_DIR / (
        f"Group_Composite_TimeSeries_KSpecific_RealMonth_"
        f"{safe_name(detection_method)}_K{k_value}_"
        f"{safe_name(direction_label)}_"
        f"StartMinus{COMPOSITE_PRE_START_MONTHS}_"
        f"EndPlus{COMPOSITE_POST_END_MONTHS}_"
        f"3MonthRollingZ.png"
    )

    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved K-specific real-month composite figure: {out_png}")


def plot_all_real_month_grids(summary, event_values, plot_group_order):
    for detection_method in DETECTION_METHODS_TO_RUN:
        for k_value in K_LIST:
            for direction in DIRECTIONS:
                plot_one_real_month_grid(
                    summary=summary,
                    event_values=event_values,
                    detection_method=detection_method,
                    k_value=k_value,
                    direction=direction,
                    plot_group_order=plot_group_order
                )


# ============================================================
# ALL-K LANDMARK PLOTS
# ============================================================

def plot_one_landmark_grid(summary, event_values, detection_method, direction, plot_group_order, landmark_cfg):
    if summary.empty or event_values.empty:
        print(f"No landmark data for {landmark_cfg['name']}. Skipping plot.")
        return

    summary_sub = summary.loc[
        (summary["detection_method"] == detection_method)
        & (summary["direction"] == direction)
    ].copy()

    if summary_sub.empty:
        print(
            f"No landmark summary data for {detection_method}, {direction}, {landmark_cfg['name']}. Skipping plot.")
        return

    n_events = get_landmark_event_count(
        event_values=event_values,
        detection_method=detection_method,
        direction=direction
    )

    ncols = 3
    nrows = int(np.ceil(len(plot_group_order) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=True
    )

    axes = np.atleast_1d(axes).ravel()

    y_min, y_max = get_y_limits_for_plot(summary_sub)

    handles = []
    labels = []

    landmark_labels = landmark_cfg["labels"]
    landmark_order_map = get_landmark_order_map(landmark_cfg)
    landmark_x_map = get_landmark_x_position_map(landmark_cfg)

    x_positions = [landmark_x_map[label] for label in landmark_labels]

    start_x = landmark_x_map.get("Start", None)
    end_x = landmark_x_map.get("End", None)

    start_segment_labels = [
        label for label in landmark_labels
        if landmark_cfg["offsets"][label][0] == "start"
    ]

    end_segment_labels = [
        label for label in landmark_labels
        if landmark_cfg["offsets"][label][0] == "end"
    ]

    start_segment_x = [landmark_x_map[label] for label in start_segment_labels]
    end_segment_x = [landmark_x_map[label] for label in end_segment_labels]

    break_after_label = landmark_cfg.get("break_after_label", None)

    if break_after_label is not None:
        break_after_x = landmark_x_map[break_after_label]

        after_break_labels = [
            label for label in landmark_labels
            if landmark_x_map[label] > break_after_x
        ]

        if len(after_break_labels) > 0:
            next_after_break_x = landmark_x_map[after_break_labels[0]]
            gap_center_x = (break_after_x + next_after_break_x) / 2.0
        else:
            gap_center_x = None
    else:
        gap_center_x = None

    for ax, group_name in zip(axes, plot_group_order):
        group_sub = summary_sub.loc[
            summary_sub["group"] == group_name
        ].copy()

        # Shade only the observed Start-side and End-side transition portions.
        # Do not shade/connect continuously across the artificial landmark gap.
        if start_x is not None and len(start_segment_x) > 0:
            start_seg_min = start_x
            start_seg_max = max(start_segment_x)

            ax.axvspan(
                start_seg_min,
                start_seg_max,
                color="#bdbdbd",
                alpha=0.16,
                linewidth=0,
                zorder=0
            )

        if end_x is not None and len(end_segment_x) > 0:
            end_seg_min = min(end_segment_x)
            end_seg_max = end_x

            ax.axvspan(
                end_seg_min,
                end_seg_max,
                color="#bdbdbd",
                alpha=0.16,
                linewidth=0,
                zorder=0
            )

        if start_x is not None:
            ax.axvline(
                start_x,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                zorder=2
            )

        if end_x is not None:
            ax.axvline(
                end_x,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                zorder=2
            )

        if gap_center_x is not None:
            ax.axvline(
                gap_center_x,
                color="#999999",
                linewidth=0.8,
                linestyle=(0, (2, 3)),
                alpha=0.55,
                zorder=2
            )

        for variable in VARIABLES_TO_PLOT:
            var_sub = group_sub.loc[
                group_sub["variable"] == variable
            ].copy()

            if var_sub.empty:
                continue

            color = VARIABLE_COLORS.get(variable, None)
            label = VARIABLE_LABELS.get(variable, variable)

            # Plot Start-side and End-side separately so the line does NOT
            # connect across the gap.
            for segment_name in ["start", "end"]:
                seg_sub = var_sub.loc[
                    var_sub["landmark_segment"] == segment_name
                ].copy()

                if seg_sub.empty:
                    continue

                seg_sub = seg_sub.sort_values("landmark_order")

                x = seg_sub["landmark_label"].map(
                    landmark_x_map).values.astype(float)
                center = seg_sub["median_z"].values.astype(float)
                p25 = seg_sub["p25_z"].values.astype(float)
                p75 = seg_sub["p75_z"].values.astype(float)

                ax.fill_between(
                    x,
                    p25,
                    p75,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=1
                )

                line, = ax.plot(
                    x,
                    center,
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=3.5,
                    label=label,
                    zorder=3
                )

                if label not in labels:
                    handles.append(line)
                    labels.append(label)

        ax.axhline(
            0,
            color="black",
            linewidth=0.7,
            linestyle="-",
            alpha=0.65,
            zorder=2
        )

        ax.set_title(group_display_name(group_name), fontsize=11)

        ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(landmark_labels, rotation=45,
                           ha="right", fontsize=8)

        ax.grid(True, alpha=0.25, linewidth=0.7)

    for ax in axes[len(plot_group_order):]:
        ax.set_axis_off()

    for ax in axes:
        ax.tick_params(labelsize=8)

    for ax in axes[-3:]:
        ax.set_xlabel("Standardized whiplash transition landmark", fontsize=9)

    for row_i in range(nrows):
        axes[row_i * 3].set_ylabel("3-month rolling z-score", fontsize=9)

    direction_label = DIRECTION_LABELS.get(direction, direction)

    min_duration = landmark_cfg.get("min_transition_duration_months", 0)

    if min_duration > 0:
        filter_text = f"Transitions shorter than {min_duration} months removed; line segments are not connected across the gap"
    else:
        filter_text = "Duplicate landmark months in short-duration events counted once; line segments are not connected across the gap"

    fig.suptitle(
        f"All-K habitat and study-area landmark composite around {direction_label} whiplash transitions\n"
        f"{detection_method}, {landmark_cfg['description']}, n={n_events} transition records\n"
        f"Line = median; shading = 25th-75th percentile; gray shading = sampled transition landmarks\n"
        f"{filter_text}",
        fontsize=13,
        y=0.995
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.88])

    out_png = landmark_cfg["fig_dir"] / (
        f"Group_Composite_TimeSeries_{safe_name(landmark_cfg['name'])}_"
        f"{safe_name(detection_method)}_"
        f"{safe_name(direction_label)}_"
        f"NoLineAcrossGap_"
        f"3MonthRollingZ.png"
    )

    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved All-K landmark composite figure: {out_png}")


def plot_all_landmark_grids(summary, event_values, plot_group_order, landmark_cfg):
    for detection_method in DETECTION_METHODS_TO_RUN:
        for direction in DIRECTIONS:
            plot_one_landmark_grid(
                summary=summary,
                event_values=event_values,
                detection_method=detection_method,
                direction=direction,
                plot_group_order=plot_group_order,
                landmark_cfg=landmark_cfg
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("FIGURE 4 HABITAT + STUDY-AREA COMPOSITE TIME SERIES")
    print("=" * 80)

    print(f"\nOutput folder: {FIGURE4_OUT_DIR}")
    print(
        f"K-specific composite window: StartDate - {COMPOSITE_PRE_START_MONTHS} months through EndDate + {COMPOSITE_POST_END_MONTHS} months")
    print("All-K composite type: landmark-based with no line connection across the landmark gap")
    print(f"Rolling window: {ROLLING_WINDOW_MONTHS} months")
    print(
        f"Z-score baseline: {ZSCORE_BASELINE_START} to {ZSCORE_BASELINE_END}")
    print(f"Detection methods: {DETECTION_METHODS_TO_RUN}")
    print(f"K values: {K_LIST}")
    print(f"Variables: {VARIABLES_TO_PLOT}")

    for landmark_cfg in ALLK_LANDMARK_ANALYSES:
        print(f"\nLandmark analysis: {landmark_cfg['name']}")
        print(f"  {landmark_cfg['description']}")
        print(f"  Folder: {landmark_cfg['folder']}")

    months_load = get_composite_load_months()

    print("\nMonths loaded for composites:")
    print(f"  {months_load[0]} to {months_load[-1]}")
    print(f"  n = {len(months_load)} months")

    print("\nLoading study boundary...")
    soc = gpd.read_file(SOCAL_SHP)

    print("\nLoading NDVI base for template...")
    ndvi_base = load_ndvi_base(GIMMS_PATH, soc)

    template_2d = ndvi_base.isel(time=0)
    template_2d = ensure_spatial_dims(template_2d)

    static_layers = prepare_static_habitat_layers(template_2d)
    plot_group_order = static_layers["plot_group_order"]

    print("\nLoading and preparing composite cubes...")

    ppt_cube = load_prism_composite_cube(
        folder=PPT_DIR,
        template_2d=template_2d,
        months_load=months_load,
        var_name="PPT"
    )

    smrz_cube = load_smrz_composite_cube(
        smrz_dir=SMRZ_DIR,
        template_2d=template_2d,
        clip_gdf=soc,
        months_load=months_load
    )

    ndvi_cube = load_ndvi_composite_cube(
        ndvi_base=ndvi_base,
        months_load=months_load
    )

    cube_by_variable = {
        "PPT": ppt_cube,
        "SMRz": smrz_cube,
        "NDVI": ndvi_cube,
    }

    ts_df = build_group_timeseries_table(
        cube_by_variable=cube_by_variable,
        static_layers=static_layers
    )

    del ppt_cube
    del smrz_cube
    del ndvi_cube
    gc.collect()

    real_month_event_values = build_real_month_composite_event_values(
        ts_df=ts_df,
        plot_group_order=plot_group_order
    )

    real_month_summary = summarize_real_month_composites(
        real_month_event_values)

    plot_all_real_month_grids(
        summary=real_month_summary,
        event_values=real_month_event_values,
        plot_group_order=plot_group_order
    )

    for landmark_cfg in ALLK_LANDMARK_ANALYSES:
        print("\n" + "#" * 80)
        print(f"RUNNING ALL-K LANDMARK ANALYSIS: {landmark_cfg['name']}")
        print("#" * 80)

        landmark_event_values = build_landmark_composite_event_values(
            ts_df=ts_df,
            plot_group_order=plot_group_order,
            landmark_cfg=landmark_cfg
        )

        landmark_summary = summarize_landmark_composites(
            event_values=landmark_event_values,
            landmark_cfg=landmark_cfg
        )

        plot_all_landmark_grids(
            summary=landmark_summary,
            event_values=landmark_event_values,
            plot_group_order=plot_group_order,
            landmark_cfg=landmark_cfg
        )

    print("\nDONE - FIGURE 4 HABITAT + STUDY-AREA COMPOSITE TIME SERIES")


if __name__ == "__main__":
    main()
