# ============================================================
# PPT-LEADING LAGGED CORRELATION ANALYSIS
#
# Produces:
# 1. Main maps:
#    p < 0.05 + |R| >= 0.30, any sign
#
# 2. Supplement maps:
#    p < 0.05 only, any sign
#
# 3. Sensitivity maps:
#    p < 0.05 + |R| >= 0.25 and 0.40, any sign
#
# 4. Diagnostic maps:
#    all valid pixels, any-sign absolute selection
#
# 5. Diagnostic CSV summaries:
#    expected-sign vs opposite-sign percentages before and after main filters
#    correlation-strength bins before and after main filters
#    lag-month percentage summaries similar to Wang et al. Figure 9
#
# Notes:
# - lag 0 = same month
# - lag 1 = PPT leads response variable by 1 month
# - lag 6 = PPT leads response variable by 6 months
# - raw statistics and GeoTIFFs remain unsmoothed
# - optional smoothed display maps are visualization only
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

from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import spearmanr
from scipy.ndimage import convolve, generic_filter
from dateutil.relativedelta import relativedelta
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

BASE_OUT = r"D:\Graduate Research\Final_Paper_Analysis"

# New finalized 3-month precipitation whiplash detections, SD = 0.8.
# Exact K-specific filenames created by the three-method detection script.
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
COASTAL_ZONE_SHP = r"D:\Graduate Research\Graduate Research\Study Area Extent\ca_state (1)\CA_State.shp"
CITIES = {
    "Los Angeles": (-118.2437, 34.0522),
    "San Diego": (-117.1611, 32.7157),
    "Santa Barbara": (-119.6982, 34.4208),
    "Palm Springs": (-116.5453, 33.8303),
    "Lancaster": (-118.1542, 34.6868),
    "Oxnard": (-119.1771, 34.1975),
}

PRISM = {
    "TMean": r"D:\Graduate Research\Graduate Research\Data\PRISM_tmean_stable_4kmM3_1981_2024",
    "TMax": r"D:\Graduate Research\Graduate Research\Data\PRISM_tempmax_stable_4kmM3_198101_202409_bil",
    "TMin": r"D:\Graduate Research\Graduate Research\Data\PRISM_tempmin_stable_4kmM3_198101_202409_bil",
    "VPDMax": r"D:\Graduate Research\Graduate Research\Data\PRISM_vpdmax_stable_4kmM3_198101_202409_bil",
    "VPDMin": r"D:\Graduate Research\Graduate Research\Data\PRISM_vpdmin_stable_4kmM3_198101_202409_bil",
}

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

K_LIST = [3, 6, 12, 24]

WINDOWS = {
    "3-Month": 3,
}

DIRECTIONS = [
    "Dry->Wet",
    "Wet->Dry",
    "Combined",
]

BUFFER_MONTHS = 3

MAX_LAG = 6
LAGS = list(range(0, MAX_LAG + 1))

MIN_SAMPLES = 6
ALPHA = 0.05

ANALYSIS_START = "1981-01"
ANALYSIS_END = "2024-12"

# Options: "zscore", "anomaly", "raw"
# zscore is recommended because it removes monthly seasonality and standardizes.
CLIMATE_TRANSFORM = "zscore"

TEST_MODE = False
TEST_MAX_ROWS = 200
TEST_MAX_COLS = 200

SAVE_RASTERS = True
SAVE_INDIVIDUAL_MAPS = True
SAVE_GRIDDED_MAPS = True
SAVE_HABITAT_BAR_PLOTS = True
SAVE_DIAGNOSTIC_BAR_PLOTS = True

# Visualization only. Raw statistics and rasters are not smoothed.
SAVE_SMOOTHED_VISUAL_MAPS = True

HABITAT_ORDER = list(VEG_SHPS.keys())

DRIVER_SPECS = [
    ("NDVI", "ndvi"),
    ("SMRz", "smrz"),
    ("TMean", PRISM["TMean"]),
    ("TMax", PRISM["TMax"]),
    ("TMin", PRISM["TMin"]),
    ("VPDMax", PRISM["VPDMax"]),
    ("VPDMin", PRISM["VPDMin"]),
]

PLOT_VARIABLE_ORDER = [v for v, _ in DRIVER_SPECS]

# Variable-specific expected signs.
# These are now used only for diagnostic summaries.
# They are not used to remove pixels from the main maps.
EXPECTED_SIGN_BY_VARIABLE = {
    "NDVI": "positive",
    "SMRz": "positive",
    "TMean": "negative",
    "TMax": "negative",
    "TMin": "negative",
    "VPDMax": "negative",
    "VPDMin": "negative",
}

# Map modes:
# - No mode removes pixels because of expected-sign behavior.
# - Main uses p < 0.05 and |R| >= 0.30, any sign.
# - Supplement uses p < 0.05 only, any sign.
# - Sensitivity uses p < 0.05 and |R| >= 0.25 or 0.40, any sign.
# - Diagnostic shows all valid pixels selected by strongest absolute correlation.
MAP_MODES = {
    "Main_AnySign_R030": {
        "label": "any sign, p < 0.05, abs(R) >= 0.30",
        "expected_sign": False,
        "p_filter": True,
        "r_threshold": 0.30,
        "is_main": True,
        "is_diagnostic": False,
    },

    "Supplement_POnly_AnySign": {
        "label": "p < 0.05 only, any sign",
        "expected_sign": False,
        "p_filter": True,
        "r_threshold": None,
        "is_main": False,
        "is_diagnostic": False,
    },

    "Sensitivity_AnySign_R025": {
        "label": "any sign, p < 0.05, abs(R) >= 0.25",
        "expected_sign": False,
        "p_filter": True,
        "r_threshold": 0.25,
        "is_main": False,
        "is_diagnostic": False,
    },

    "Sensitivity_AnySign_R040": {
        "label": "any sign, p < 0.05, abs(R) >= 0.40",
        "expected_sign": False,
        "p_filter": True,
        "r_threshold": 0.40,
        "is_main": False,
        "is_diagnostic": False,
    },

    "Diagnostic_AllPixels_AnySign": {
        "label": "diagnostic, all valid pixels, any sign",
        "expected_sign": False,
        "p_filter": False,
        "r_threshold": None,
        "is_main": False,
        "is_diagnostic": True,
    },
}

MAIN_MODE_KEY = "Main_AnySign_R030"
DIAGNOSTIC_MODE_KEY = "Diagnostic_AllPixels_AnySign"

# Smoothed display maps are only created for these modes.
SMOOTHED_MODE_KEYS = [
    MAIN_MODE_KEY,
]


# ============================================================
# COLOR MAPS
# ============================================================

CORR_CMAP = plt.get_cmap("RdYlBu_r").copy()
CORR_CMAP.set_bad("white")

LAG_COLORS = [
    "#08306b",  # 0 month
    "#2171b5",  # 1 month
    "#41b6c4",  # 2 months
    "#7fcdbb",  # 3 months
    "#c7e9b4",  # 4 months
    "#fdae61",  # 5 months
    "#d7191c",  # 6 months
]

LAG_CMAP = ListedColormap(LAG_COLORS)
LAG_CMAP.set_bad("white")
LAG_NORM = BoundaryNorm(np.arange(-0.5, MAX_LAG + 1.5, 1), LAG_CMAP.N)

BAR_LAG_COLORS = {
    0: "#08306b",
    1: "#2171b5",
    2: "#41b6c4",
    3: "#00cc44",
    4: "#ff7f00",
    5: "#a65628",
    6: "#000000",
}


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


def month_range(start_ym, end_ym):
    return pd.period_range(
        pd.Period(start_ym, freq="M"),
        pd.Period(end_ym, freq="M"),
        freq="M"
    ).strftime("%Y-%m").tolist()


def expand_event_months(date):
    return pd.period_range(
        date - relativedelta(months=BUFFER_MONTHS),
        date + relativedelta(months=BUFFER_MONTHS),
        freq="M"
    ).strftime("%Y-%m")


def get_event_months(df):
    months = set()

    for _, row in df.iterrows():
        months |= set(expand_event_months(row["StartDate"]))
        months |= set(expand_event_months(row["EndDate"]))

    months = sorted(months)

    months = [
        m for m in months
        if ANALYSIS_START <= m <= ANALYSIS_END
    ]

    return months


def add_load_buffer_months(base_months, win_len, max_lag):
    expanded = set()
    back = max(0, win_len - 1)

    for ym in base_months:
        p = pd.Period(ym, freq="M")

        for offset in range(-back, max_lag + 1):
            expanded.add((p + offset).strftime("%Y-%m"))

    return sorted(expanded)


def get_full_analysis_load_months(win_len, max_lag):
    base_months = month_range(ANALYSIS_START, ANALYSIS_END)
    return add_load_buffer_months(base_months, win_len, max_lag)


def period_add(ym, lag):
    return (pd.Period(ym, freq="M") + lag).strftime("%Y-%m")


def subset_by_ym(da, months_keep):
    mask = np.isin(da["ym"].values, months_keep)
    return da.isel(time=mask)


def get_datetime_index(da):
    idx = da.indexes["time"]

    if hasattr(idx, "to_datetimeindex"):
        return idx.to_datetimeindex()

    return pd.to_datetime(idx)


def standardize_monthly_time(da):
    """
    Converts time stamps to month-start.
    If multiple observations occur in the same month, they are averaged.
    This avoids blank NDVI maps caused by mid-month or non-month-start timestamps.
    """
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


def reindex_monthly_for_rolling(cube, win_len):
    cube = standardize_monthly_time(cube)

    time_vals = pd.to_datetime(cube["time"].values)

    if len(time_vals) == 0:
        return cube

    start = pd.Timestamp(time_vals.min()).to_period("M").to_timestamp()
    end = pd.Timestamp(time_vals.max()).to_period("M").to_timestamp()

    full_time = pd.date_range(start, end, freq="MS")

    cube = cube.reindex(time=full_time)

    cube = cube.assign_coords(
        ym=("time", pd.DatetimeIndex(full_time).strftime("%Y-%m"))
    )

    if win_len is not None and win_len > 1:
        cube = cube.rolling(time=win_len, min_periods=win_len).mean()

    cube = standardize_monthly_time(cube)

    return cube


def apply_monthly_transform(cube, transform_mode):
    cube = standardize_monthly_time(cube)

    if transform_mode == "raw":
        return cube

    month_group = cube.groupby("time.month")

    clim_mean = month_group.mean("time", skipna=True)
    anomaly = month_group - clim_mean

    if transform_mode == "anomaly":
        return anomaly

    if transform_mode == "zscore":
        clim_std = month_group.std("time", skipna=True)
        clim_std = clim_std.where(clim_std > 0)
        z = anomaly.groupby("time.month") / clim_std
        return z

    raise ValueError(f"Unknown CLIMATE_TRANSFORM: {transform_mode}")


def prepare_cube_for_correlation(cube, win_len, transform_mode, months_keep=None):
    cube = standardize_monthly_time(cube)
    cube = reindex_monthly_for_rolling(cube, win_len)
    cube = apply_monthly_transform(cube, transform_mode)
    cube = standardize_monthly_time(cube)

    if months_keep is not None:
        cube = subset_by_ym(cube, months_keep)

    cube = cube.astype("float32").load()

    return cube


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


def clean_geometries(gdf):
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    return gdf


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
    print("Preparing habitat layer using pixel-center joins...")

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

    habitat_order = [
        h for h in HABITAT_ORDER
        if h in static["habitat"].dropna().unique()
    ]

    print("\nHabitat counts assigned to grid pixel centers:")
    print(static["habitat"].value_counts(dropna=False))

    return {
        "static_pixel_table": static.drop(columns="geometry"),
        "habitat_order": habitat_order,
    }


# ============================================================
# SMOOTHING FOR VISUALIZATION ONLY
# ============================================================

def smooth_continuous_da_for_display(da):
    arr = da.values.astype(float)

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=float)

    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0)

    numerator = convolve(
        arr_filled,
        kernel,
        mode="constant",
        cval=0.0
    )

    denominator = convolve(
        valid.astype(float),
        kernel,
        mode="constant",
        cval=0.0
    )

    out = np.full_like(arr, np.nan, dtype=float)

    good = denominator > 0
    out[good] = numerator[good] / denominator[good]

    # Do not expand into originally blank areas.
    out[~valid] = np.nan

    out_da = xr.DataArray(
        out.astype("float32"),
        coords=da.coords,
        dims=da.dims,
        name=da.name
    )

    out_da = out_da.rio.write_crs(da.rio.crs, inplace=False)
    out_da = ensure_spatial_dims(out_da)

    return out_da


def mode_filter_lag_window(values):
    vals = values[np.isfinite(values)]

    if len(vals) == 0:
        return np.nan

    vals = vals.astype(int)
    vals = vals[(vals >= 0) & (vals <= MAX_LAG)]

    if len(vals) == 0:
        return np.nan

    counts = np.bincount(vals, minlength=MAX_LAG + 1)

    return float(np.argmax(counts))


def smooth_lag_da_for_display(da):
    arr = da.values.astype(float)
    valid = np.isfinite(arr)

    filtered = generic_filter(
        arr,
        function=mode_filter_lag_window,
        size=3,
        mode="constant",
        cval=np.nan
    )

    filtered[~valid] = np.nan

    out_da = xr.DataArray(
        filtered.astype("float32"),
        coords=da.coords,
        dims=da.dims,
        name=da.name
    )

    out_da = out_da.rio.write_crs(da.rio.crs, inplace=False)
    out_da = ensure_spatial_dims(out_da)

    return out_da


def get_display_da(da, map_type, smooth_display=False):
    if not smooth_display:
        return da

    if map_type == "correlation":
        return smooth_continuous_da_for_display(da)

    if map_type == "lag":
        return smooth_lag_da_for_display(da)

    return da


# ============================================================
# DATA LOADERS
# ============================================================

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


def load_ndvi_driver(ndvi_base, months_load, win_len):
    ndvi = subset_by_ym(ndvi_base, months_load)

    ndvi = prepare_cube_for_correlation(
        cube=ndvi,
        win_len=win_len,
        transform_mode=CLIMATE_TRANSFORM,
        months_keep=months_load
    )

    print_cube_diagnostics(ndvi, "NDVI driver after transform")

    return ndvi


def load_prism_cube(folder, template_2d, months_load, win_len, var_name="PRISM"):
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

    cube = prepare_cube_for_correlation(
        cube=cube,
        win_len=win_len,
        transform_mode=CLIMATE_TRANSFORM,
        months_keep=months_load
    )

    print_cube_diagnostics(cube, var_name)

    return cube


def load_smrz_cube(smrz_dir, template_2d, clip_gdf, months_load, win_len):
    print("Loading SMRz, memory-safe...")

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

    # If there is an extra dimension, select the first layer.
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

    cube = prepare_cube_for_correlation(
        cube=cube,
        win_len=win_len,
        transform_mode=CLIMATE_TRANSFORM,
        months_keep=months_load
    )

    print_cube_diagnostics(cube, "SMRz driver after transform")

    return cube


# ============================================================
# LAGGED CORRELATION
# ============================================================

def get_lag_indices(ppt_cube, response_cube, base_months, lag):
    ppt_index = {
        ym: i for i, ym in enumerate(ppt_cube["ym"].values)
    }

    response_index = {
        ym: i for i, ym in enumerate(response_cube["ym"].values)
    }

    ppt_idx = []
    response_idx = []

    for ym in base_months:
        response_ym = period_add(ym, lag)

        if ym in ppt_index and response_ym in response_index:
            ppt_idx.append(ppt_index[ym])
            response_idx.append(response_index[response_ym])

    return np.array(ppt_idx), np.array(response_idx)


def spearman_arrays(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < MIN_SAMPLES:
        return np.nan, np.nan, int(mask.sum())

    x_valid = x[mask]
    y_valid = y[mask]

    if len(np.unique(x_valid)) <= 1 or len(np.unique(y_valid)) <= 1:
        return np.nan, np.nan, int(mask.sum())

    r, p = spearmanr(x_valid, y_valid)

    return float(r), float(p), int(mask.sum())


def is_expected_sign(r, variable):
    mode = EXPECTED_SIGN_BY_VARIABLE.get(variable, "absolute")

    if mode == "positive":
        return r > 0

    if mode == "negative":
        return r < 0

    if mode == "absolute":
        return True

    raise ValueError(f"Unknown expected sign mode: {mode}")


def candidate_passes_mode(r, p, variable, mode_spec):
    if not np.isfinite(r) or not np.isfinite(p):
        return False

    if mode_spec["p_filter"] and p > ALPHA:
        return False

    if mode_spec["r_threshold"] is not None:
        if abs(r) < float(mode_spec["r_threshold"]):
            return False

    if mode_spec["expected_sign"]:
        if not is_expected_sign(r, variable):
            return False

    return True


def candidate_score(r, variable, mode_spec):
    if mode_spec["expected_sign"]:
        if not is_expected_sign(r, variable):
            return -np.inf

    return abs(r)


def make_blank_da_like(template_da, name):
    base = template_da.copy(deep=False)
    arr = np.full(base.shape, np.nan, dtype=np.float32)

    da = xr.DataArray(
        arr,
        coords=base.coords,
        dims=base.dims,
        name=name
    )

    da = da.rio.write_crs(base.rio.crs, inplace=False)
    da = ensure_spatial_dims(da)

    return da


def compute_best_lag_maps(ppt_cube, response_cube, base_months, variable):
    common_crs = ppt_cube.rio.crs

    base = ppt_cube.isel(time=0)
    base = ensure_spatial_dims(base)

    if TEST_MODE:
        row_n = min(base.shape[0], TEST_MAX_ROWS)
        col_n = min(base.shape[1], TEST_MAX_COLS)

        base = base.isel(y=slice(0, row_n), x=slice(0, col_n))
        ppt_cube = ppt_cube.isel(y=slice(0, row_n), x=slice(0, col_n))
        response_cube = response_cube.isel(
            y=slice(0, row_n), x=slice(0, col_n))

    nrows, ncols = base.shape

    # Initialize arrays for all map modes.
    arrays = {}

    for mode_key in MAP_MODES.keys():
        arrays[mode_key] = {
            "corr": np.full((nrows, ncols), np.nan, dtype=np.float32),
            "lag": np.full((nrows, ncols), np.nan, dtype=np.float32),
            "p": np.full((nrows, ncols), np.nan, dtype=np.float32),
            "n": np.full((nrows, ncols), np.nan, dtype=np.float32),
            "score": np.full((nrows, ncols), -np.inf, dtype=np.float32),
        }

    lag_pairs = {}

    for lag in LAGS:
        ppt_idx, resp_idx = get_lag_indices(
            ppt_cube,
            response_cube,
            base_months,
            lag
        )

        if len(ppt_idx) >= MIN_SAMPLES:
            lag_pairs[lag] = (ppt_idx, resp_idx)
            print(f"    Lag {lag}: paired months = {len(ppt_idx)}")
        else:
            print(f"    Lag {lag}: not enough paired months ({len(ppt_idx)}).")

    if len(lag_pairs) == 0:
        raise ValueError("No valid lag pairs were available.")

    ppt_values = ppt_cube.values
    response_values = response_cube.values

    for i in range(nrows):
        if i % 50 == 0:
            print(f"    Row {i + 1} / {nrows}")

        for j in range(ncols):
            for lag, (ppt_idx, resp_idx) in lag_pairs.items():
                x = ppt_values[ppt_idx, i, j]
                y = response_values[resp_idx, i, j]

                r, p, n = spearman_arrays(x, y)

                if not np.isfinite(r) or not np.isfinite(p):
                    continue

                for mode_key, mode_spec in MAP_MODES.items():
                    if not candidate_passes_mode(r, p, variable, mode_spec):
                        continue

                    score = candidate_score(r, variable, mode_spec)

                    if score > arrays[mode_key]["score"][i, j]:
                        arrays[mode_key]["score"][i, j] = score
                        arrays[mode_key]["corr"][i, j] = r
                        arrays[mode_key]["lag"][i, j] = lag
                        arrays[mode_key]["p"][i, j] = p
                        arrays[mode_key]["n"][i, j] = n

    def make_da(arr, name):
        da = xr.DataArray(
            arr,
            coords=base.coords,
            dims=base.dims,
            name=name
        )

        da = da.rio.write_crs(common_crs, inplace=False)
        da = ensure_spatial_dims(da)

        return da

    out = {}

    for mode_key in MAP_MODES.keys():
        out[mode_key] = {
            "corr": make_da(
                arrays[mode_key]["corr"],
                f"{variable}_{mode_key}_spearman_r"
            ),
            "lag": make_da(
                arrays[mode_key]["lag"],
                f"{variable}_{mode_key}_response_lag_months"
            ),
            "p": make_da(
                arrays[mode_key]["p"],
                f"{variable}_{mode_key}_p_value"
            ),
            "n": make_da(
                arrays[mode_key]["n"],
                f"{variable}_{mode_key}_n_month_pairs"
            ),
        }

    return out


def summarize_map_outputs(map_result_by_mode, var_name):
    records = []

    for mode_key, mode_result in map_result_by_mode.items():
        corr = mode_result["corr"].values
        pvals = mode_result["p"].values

        records.append({
            "variable": var_name,
            "mode": mode_key,
            "mode_label": MAP_MODES[mode_key]["label"],
            "expected_sign": EXPECTED_SIGN_BY_VARIABLE.get(var_name, "absolute"),
            "mapped_pixels": int(np.isfinite(corr).sum()),
            "min_best_r": float(np.nanmin(corr)) if np.isfinite(corr).any() else np.nan,
            "max_best_r": float(np.nanmax(corr)) if np.isfinite(corr).any() else np.nan,
            "median_best_r": float(np.nanmedian(corr)) if np.isfinite(corr).any() else np.nan,
            "min_best_p": float(np.nanmin(pvals)) if np.isfinite(pvals).any() else np.nan,
            "median_best_p": float(np.nanmedian(pvals)) if np.isfinite(pvals).any() else np.nan,
            "alpha": ALPHA,
            "r_threshold": MAP_MODES[mode_key]["r_threshold"],
            "uses_expected_sign": MAP_MODES[mode_key]["expected_sign"],
            "uses_p_filter": MAP_MODES[mode_key]["p_filter"],
        })

    return pd.DataFrame.from_records(records)


# ============================================================
# SIGN, CORRELATION-BIN, AND LAG DIAGNOSTICS
# ============================================================

def classify_sign_category(r_values, variable):
    expected_mode = EXPECTED_SIGN_BY_VARIABLE.get(variable, "absolute")
    r_values = np.asarray(r_values, dtype=float)

    out = np.full(r_values.shape, "no_expected_sign_defined", dtype=object)

    if expected_mode == "positive":
        out[r_values > 0] = "expected_sign"
        out[r_values < 0] = "opposite_sign"
        out[r_values == 0] = "zero"

    elif expected_mode == "negative":
        out[r_values < 0] = "expected_sign"
        out[r_values > 0] = "opposite_sign"
        out[r_values == 0] = "zero"

    else:
        out[np.isfinite(r_values)] = "not_classified"

    out[~np.isfinite(r_values)] = "no_data"

    return out


def make_sign_diagnostic_records(map_result_by_mode, variable, direction, win_label):
    records = []

    comparison_modes = [
        (DIAGNOSTIC_MODE_KEY, "before_main_filters_all_valid_pixels"),
        (MAIN_MODE_KEY, "after_main_filters_p005_absR030"),
    ]

    for mode_key, filter_stage in comparison_modes:
        if mode_key not in map_result_by_mode:
            continue

        corr = map_result_by_mode[mode_key]["corr"].values
        valid = np.isfinite(corr)
        total = int(valid.sum())

        if total == 0:
            for category in ["expected_sign", "opposite_sign", "zero"]:
                records.append({
                    "direction": direction,
                    "window": win_label,
                    "variable": variable,
                    "filter_stage": filter_stage,
                    "mode": mode_key,
                    "expected_sign_definition": EXPECTED_SIGN_BY_VARIABLE.get(variable, "absolute"),
                    "sign_category": category,
                    "pixel_count": 0,
                    "percent_of_stage_pixels": np.nan,
                    "total_stage_pixels": 0,
                })
            continue

        categories = classify_sign_category(corr, variable)

        for category in ["expected_sign", "opposite_sign", "zero", "not_classified"]:
            count = int(((categories == category) & valid).sum())

            if count == 0 and category == "not_classified":
                continue

            records.append({
                "direction": direction,
                "window": win_label,
                "variable": variable,
                "filter_stage": filter_stage,
                "mode": mode_key,
                "expected_sign_definition": EXPECTED_SIGN_BY_VARIABLE.get(variable, "absolute"),
                "sign_category": category,
                "pixel_count": count,
                "percent_of_stage_pixels": 100.0 * count / total if total > 0 else np.nan,
                "total_stage_pixels": total,
            })

    return pd.DataFrame.from_records(records)


def make_correlation_bin_records(map_result_by_mode, variable, direction, win_label):
    records = []

    bin_edges = [-1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    bin_labels = [
        "[-1.0,-0.7)",
        "[-0.7,-0.5)",
        "[-0.5,-0.3)",
        "[-0.3,-0.1)",
        "[-0.1,0.0)",
        "[0.0,0.1)",
        "[0.1,0.3)",
        "[0.3,0.5)",
        "[0.5,0.7)",
        "[0.7,1.0]",
    ]

    comparison_modes = [
        (DIAGNOSTIC_MODE_KEY, "before_main_filters_all_valid_pixels"),
        (MAIN_MODE_KEY, "after_main_filters_p005_absR030"),
    ]

    for mode_key, filter_stage in comparison_modes:
        if mode_key not in map_result_by_mode:
            continue

        corr = map_result_by_mode[mode_key]["corr"].values
        valid_corr = corr[np.isfinite(corr)]
        total = int(valid_corr.size)

        if total == 0:
            for label in bin_labels:
                records.append({
                    "direction": direction,
                    "window": win_label,
                    "variable": variable,
                    "filter_stage": filter_stage,
                    "mode": mode_key,
                    "correlation_bin": label,
                    "pixel_count": 0,
                    "percent_of_stage_pixels": np.nan,
                    "total_stage_pixels": 0,
                })
            continue

        # Include the right edge for the final bin.
        binned = pd.cut(
            valid_corr,
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False
        )

        # Manually place exact 1.0 values in the final bin.
        binned = pd.Series(binned).astype(object)
        binned[pd.Series(valid_corr) == 1.0] = bin_labels[-1]

        for label in bin_labels:
            count = int((binned == label).sum())

            records.append({
                "direction": direction,
                "window": win_label,
                "variable": variable,
                "filter_stage": filter_stage,
                "mode": mode_key,
                "correlation_bin": label,
                "pixel_count": count,
                "percent_of_stage_pixels": 100.0 * count / total if total > 0 else np.nan,
                "total_stage_pixels": total,
            })

    return pd.DataFrame.from_records(records)


def make_lag_distribution_records(map_result_by_mode, variable, direction, win_label):
    records = []

    comparison_modes = [
        (DIAGNOSTIC_MODE_KEY, "before_main_filters_all_valid_pixels"),
        (MAIN_MODE_KEY, "after_main_filters_p005_absR030"),
    ]

    for mode_key, filter_stage in comparison_modes:
        if mode_key not in map_result_by_mode:
            continue

        lag = map_result_by_mode[mode_key]["lag"].values
        corr = map_result_by_mode[mode_key]["corr"].values

        valid = np.isfinite(lag) & np.isfinite(corr)
        total = int(valid.sum())
        categories = classify_sign_category(corr, variable)

        for lag_value in LAGS:
            lag_mask = valid & (lag == lag_value)
            lag_count = int(lag_mask.sum())

            records.append({
                "direction": direction,
                "window": win_label,
                "variable": variable,
                "filter_stage": filter_stage,
                "mode": mode_key,
                "lag_months": int(lag_value),
                "sign_category": "all_signs",
                "pixel_count": lag_count,
                "percent_of_stage_pixels": 100.0 * lag_count / total if total > 0 else np.nan,
                "total_stage_pixels": total,
            })

            for sign_category in ["expected_sign", "opposite_sign", "zero"]:
                sign_mask = valid & (categories == sign_category)
                sign_total = int(sign_mask.sum())
                count = int((lag_mask & sign_mask).sum())

                records.append({
                    "direction": direction,
                    "window": win_label,
                    "variable": variable,
                    "filter_stage": filter_stage,
                    "mode": mode_key,
                    "lag_months": int(lag_value),
                    "sign_category": sign_category,
                    "pixel_count": count,
                    "percent_of_stage_pixels": 100.0 * count / total if total > 0 else np.nan,
                    "percent_of_sign_category_pixels": 100.0 * count / sign_total if sign_total > 0 else np.nan,
                    "total_stage_pixels": total,
                    "total_sign_category_pixels": sign_total,
                })

    return pd.DataFrame.from_records(records)


def make_all_diagnostic_tables(map_result_by_mode, variable, direction, win_label):
    sign_df = make_sign_diagnostic_records(
        map_result_by_mode=map_result_by_mode,
        variable=variable,
        direction=direction,
        win_label=win_label,
    )

    bin_df = make_correlation_bin_records(
        map_result_by_mode=map_result_by_mode,
        variable=variable,
        direction=direction,
        win_label=win_label,
    )

    lag_df = make_lag_distribution_records(
        map_result_by_mode=map_result_by_mode,
        variable=variable,
        direction=direction,
        win_label=win_label,
    )

    return sign_df, bin_df, lag_df


# ============================================================
# DIAGNOSTIC BAR PLOTS
# ============================================================

def clean_stage_label(stage):
    stage = str(stage)

    if stage == "before_main_filters_all_valid_pixels":
        return "Before filters\nall valid pixels"

    if stage == "after_main_filters_p005_absR030":
        return "After filters\np < 0.05 and |R| >= 0.30"

    return stage.replace("_", " ")


def save_sign_summary_barplots(sign_df, out_dir, direction, win_label):
    if sign_df is None or sign_df.empty:
        return

    plot_dir = Path(out_dir) / "Diagnostic_Barplots" / \
        "Expected_vs_Opposite_Sign"
    plot_dir.mkdir(parents=True, exist_ok=True)

    categories = ["expected_sign", "opposite_sign", "zero", "not_classified"]
    category_labels = {
        "expected_sign": "Expected sign",
        "opposite_sign": "Opposite sign",
        "zero": "Zero",
        "not_classified": "Not classified",
    }

    variables = [
        v for v in PLOT_VARIABLE_ORDER if v in sign_df["variable"].unique()]
    stages = [
        "before_main_filters_all_valid_pixels",
        "after_main_filters_p005_absR030",
    ]

    # --------------------------------------------------------
    # Stacked percentage plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(7.2 * len(stages), 5.5),
        sharey=True
    )

    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        sub = sign_df.loc[sign_df["filter_stage"] == stage].copy()
        bottoms = np.zeros(len(variables), dtype=float)
        x = np.arange(len(variables))

        for category in categories:
            vals = []

            for variable in variables:
                row = sub.loc[
                    (sub["variable"] == variable)
                    & (sub["sign_category"] == category)
                ]

                if row.empty:
                    vals.append(0.0)
                else:
                    vals.append(float(row["percent_of_stage_pixels"].iloc[0]))

            vals = np.asarray(vals, dtype=float)
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=category_labels.get(category, category)
            )
            bottoms += vals

        ax.set_title(clean_stage_label(stage))
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha="right")
        ax.set_ylabel("Percent of mapped pixels")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(labels), frameon=False)
    fig.suptitle(
        f"Expected-sign vs opposite-sign correlation pixels\n{direction}, {win_label}",
        y=1.04,
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = plot_dir / \
        f"Expected_vs_Opposite_Sign_Percent_{safe_name(direction)}_{safe_name(win_label)}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expected/opposite sign percentage bar plot: {out_png}")

    # --------------------------------------------------------
    # Retained and removed pixel-count plot
    # --------------------------------------------------------
    retained_records = []

    for variable in variables:
        before = sign_df.loc[
            (sign_df["variable"] == variable)
            & (sign_df["filter_stage"] == "before_main_filters_all_valid_pixels")
        ]
        after = sign_df.loc[
            (sign_df["variable"] == variable)
            & (sign_df["filter_stage"] == "after_main_filters_p005_absR030")
        ]

        before_total = float(
            before["total_stage_pixels"].max()) if not before.empty else 0.0
        after_total = float(
            after["total_stage_pixels"].max()) if not after.empty else 0.0
        removed = max(before_total - after_total, 0.0)
        retained_percent = 100.0 * after_total / \
            before_total if before_total > 0 else np.nan

        retained_records.append({
            "variable": variable,
            "retained_pixels": after_total,
            "removed_pixels": removed,
            "before_pixels": before_total,
            "retained_percent": retained_percent,
        })

    retained_df = pd.DataFrame(retained_records)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    x = np.arange(len(retained_df))
    retained_vals = retained_df["retained_pixels"].values
    removed_vals = retained_df["removed_pixels"].values

    ax.bar(x, retained_vals, label="Retained after filters")
    ax.bar(x, removed_vals, bottom=retained_vals, label="Removed by filters")

    for i, row in retained_df.iterrows():
        if np.isfinite(row["retained_percent"]):
            ax.text(
                i,
                row["before_pixels"] * 1.01,
                f"{row['retained_percent']:.1f}% kept",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )

    ax.set_xticks(x)
    ax.set_xticklabels(retained_df["variable"], rotation=45, ha="right")
    ax.set_ylabel("Pixel count")
    ax.set_title(
        f"Pixels retained vs removed by main filters\n{direction}, {win_label}, p < 0.05 and |R| >= 0.30"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_png = plot_dir / \
        f"Pixels_Retained_vs_Removed_{safe_name(direction)}_{safe_name(win_label)}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved retained/removed pixel-count bar plot: {out_png}")


def save_correlation_bin_barplots(bin_df, out_dir, direction, win_label):
    if bin_df is None or bin_df.empty:
        return

    plot_dir = Path(out_dir) / "Diagnostic_Barplots" / "Correlation_Bins"
    plot_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        v for v in PLOT_VARIABLE_ORDER if v in bin_df["variable"].unique()]
    stages = [
        "before_main_filters_all_valid_pixels",
        "after_main_filters_p005_absR030",
    ]

    for variable in variables:
        sub_v = bin_df.loc[bin_df["variable"] == variable].copy()
        if sub_v.empty:
            continue

        bin_order = list(sub_v["correlation_bin"].drop_duplicates())

        fig, axes = plt.subplots(
            1,
            len(stages),
            figsize=(8.2 * len(stages), 5.2),
            sharey=True
        )

        if len(stages) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            sub = sub_v.loc[sub_v["filter_stage"] == stage].copy()

            vals = []
            counts = []

            for label in bin_order:
                row = sub.loc[sub["correlation_bin"] == label]
                if row.empty:
                    vals.append(0.0)
                    counts.append(0)
                else:
                    vals.append(float(row["percent_of_stage_pixels"].iloc[0]))
                    counts.append(int(row["pixel_count"].iloc[0]))

            x = np.arange(len(bin_order))
            bars = ax.bar(x, vals)

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90
                    )

            ax.set_title(clean_stage_label(stage))
            ax.set_xticks(x)
            ax.set_xticklabels(bin_order, rotation=45, ha="right")
            ax.set_ylabel("Percent of mapped pixels")
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(
            f"Correlation-bin distribution for {variable}\n{direction}, {win_label}",
            y=1.03,
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_png = plot_dir / \
            f"Correlation_Bins_{safe_name(variable)}_{safe_name(direction)}_{safe_name(win_label)}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved correlation-bin bar plot: {out_png}")


def save_lag_distribution_barplots(lag_df, out_dir, direction, win_label):
    if lag_df is None or lag_df.empty:
        return

    plot_dir = Path(out_dir) / "Diagnostic_Barplots" / "Lag_Distributions"
    plot_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        v for v in PLOT_VARIABLE_ORDER if v in lag_df["variable"].unique()]
    stages = [
        "before_main_filters_all_valid_pixels",
        "after_main_filters_p005_absR030",
    ]

    for variable in variables:
        sub_v = lag_df.loc[
            (lag_df["variable"] == variable)
            & (lag_df["sign_category"] == "all_signs")
        ].copy()

        if sub_v.empty:
            continue

        fig, axes = plt.subplots(
            1,
            len(stages),
            figsize=(7.2 * len(stages), 5.0),
            sharey=True
        )

        if len(stages) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            sub = sub_v.loc[sub_v["filter_stage"] == stage].copy()
            vals = []
            counts = []

            for lag_value in LAGS:
                row = sub.loc[sub["lag_months"] == lag_value]
                if row.empty:
                    vals.append(0.0)
                    counts.append(0)
                else:
                    vals.append(float(row["percent_of_stage_pixels"].iloc[0]))
                    counts.append(int(row["pixel_count"].iloc[0]))

            x = np.arange(len(LAGS))
            bars = ax.bar(x, vals)

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=0
                    )

            ax.set_title(clean_stage_label(stage))
            ax.set_xticks(x)
            ax.set_xticklabels([str(lag) for lag in LAGS])
            ax.set_xlabel("PPT-leading lag, months")
            ax.set_ylabel("Percent of mapped pixels")
            ax.set_ylim(0, max(5, np.nanmax(vals) * 1.20 if len(vals) else 5))
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(
            f"Lag-month distribution for {variable}\n{direction}, {win_label}",
            y=1.03,
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_png = plot_dir / \
            f"Lag_Distribution_{safe_name(variable)}_{safe_name(direction)}_{safe_name(win_label)}.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved lag-distribution bar plot: {out_png}")


def save_diagnostic_barplots(sign_df, bin_df, lag_df, out_dir, direction, win_label):
    if not SAVE_DIAGNOSTIC_BAR_PLOTS:
        return

    save_sign_summary_barplots(
        sign_df=sign_df,
        out_dir=out_dir,
        direction=direction,
        win_label=win_label,
    )

    save_correlation_bin_barplots(
        bin_df=bin_df,
        out_dir=out_dir,
        direction=direction,
        win_label=win_label,
    )

    save_lag_distribution_barplots(
        lag_df=lag_df,
        out_dir=out_dir,
        direction=direction,
        win_label=win_label,
    )

# ============================================================
# HABITAT BAR PLOTS
# ============================================================


def get_habitat_weighted_series(cube, static_df, habitat):
    df = static_df.loc[
        (static_df["habitat"] == habitat)
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


def habitat_candidate_flags(r, p, var_name):
    flags = {}

    for mode_key, mode_spec in MAP_MODES.items():
        flags[f"significant_{mode_key}"] = bool(
            candidate_passes_mode(r, p, var_name, mode_spec)
        )

    return flags


def compute_habitat_lag_correlations(
    ppt_cube,
    response_cube,
    static_layers,
    base_months,
    var_name,
    direction,
    win_label
):
    static_df = static_layers["static_pixel_table"]
    habitat_order = static_layers["habitat_order"]

    records = []

    for habitat in habitat_order:
        ppt_series = get_habitat_weighted_series(
            ppt_cube,
            static_df,
            habitat
        )

        response_series = get_habitat_weighted_series(
            response_cube,
            static_df,
            habitat
        )

        if ppt_series.empty or response_series.empty:
            continue

        for lag in LAGS:
            x_vals = []
            y_vals = []

            for ym in base_months:
                response_ym = period_add(ym, lag)

                if ym not in ppt_series.index:
                    continue

                if response_ym not in response_series.index:
                    continue

                x_vals.append(ppt_series.loc[ym])
                y_vals.append(response_series.loc[response_ym])

            x = np.asarray(x_vals, dtype=float)
            y = np.asarray(y_vals, dtype=float)

            r, p, n = spearman_arrays(x, y)

            record = {
                "direction": direction,
                "window": win_label,
                "variable": var_name,
                "expected_sign": EXPECTED_SIGN_BY_VARIABLE.get(var_name, "absolute"),
                "habitat": habitat,
                "lag_months": lag,
                "spearman_r": r,
                "p_value": p,
                "n_month_pairs": n,
            }

            record.update(habitat_candidate_flags(r, p, var_name))

            # Convenience column for the main map setting.
            record["significant_main"] = record[f"significant_{MAIN_MODE_KEY}"]

            records.append(record)

    return pd.DataFrame.from_records(records)


def save_habitat_lag_barplots(hab_corr_df, out_dir, direction, win_label):
    if hab_corr_df.empty:
        return

    plot_dir = Path(out_dir) / "Habitat_Barplots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    variables = PLOT_VARIABLE_ORDER

    habitats = [
        h for h in HABITAT_ORDER
        if h in hab_corr_df["habitat"].unique()
    ]

    n_habitats = len(habitats)
    fig_height = max(4.5, 2.7 * n_habitats)

    fig, axes = plt.subplots(
        n_habitats,
        1,
        figsize=(16, fig_height),
        sharex=True,
        sharey=True
    )

    if n_habitats == 1:
        axes = [axes]

    x = np.arange(len(variables))
    total_width = 0.82
    bar_width = total_width / len(LAGS)

    for ax, habitat in zip(axes, habitats):
        sub_h = hab_corr_df.loc[
            hab_corr_df["habitat"] == habitat
        ].copy()

        for lag_i, lag in enumerate(LAGS):
            vals = []
            sigs = []

            for variable in variables:
                row = sub_h.loc[
                    (sub_h["variable"] == variable)
                    & (sub_h["lag_months"] == lag)
                ]

                if row.empty:
                    vals.append(np.nan)
                    sigs.append(False)
                else:
                    vals.append(float(row["spearman_r"].iloc[0]))
                    sigs.append(bool(row["significant_main"].iloc[0]))

            offset = (lag_i - (len(LAGS) - 1) / 2) * bar_width

            bars = ax.bar(
                x + offset,
                vals,
                width=bar_width,
                color=BAR_LAG_COLORS[lag],
                label=f"{lag}-month",
                edgecolor="none"
            )

            for bar, sig in zip(bars, sigs):
                if not sig:
                    bar.set_alpha(0.25)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.30, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(-0.30, color="gray", linewidth=0.5, linestyle=":")

        ax.set_ylim(-1, 1)
        ax.set_ylabel("Spearman R")
        ax.set_title(habitat, loc="left", fontsize=11)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(variables, rotation=45, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(LAGS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995)
    )

    fig.suptitle(
        f"Habitat-level lagged Spearman correlations: PPT leading response variables\n"
        f"{direction}, {win_label}, {CLIMATE_TRANSFORM} | faded bars fail main threshold: p < {ALPHA}, abs(R) >= 0.30",
        y=1.02,
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = plot_dir / \
        f"Habitat_Lagged_Correlation_Bars_{safe_name(direction)}_{safe_name(win_label)}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved habitat bar plot: {out_png}")


# ============================================================
# MAP OUTPUTS
# ============================================================

def load_optional_boundary(shp_path, layer_name):
    """
    Loads an optional boundary shapefile.
    If the file is missing or cannot be read, the script continues without stopping.
    """
    try:
        if shp_path is None or not os.path.exists(shp_path):
            print(
                f"Optional {layer_name} shapefile not found, skipping: {shp_path}")
            return None

        gdf = gpd.read_file(shp_path)

        if gdf.empty:
            print(
                f"Optional {layer_name} shapefile is empty, skipping: {shp_path}")
            return None

        if gdf.crs is None:
            print(
                f"Optional {layer_name} shapefile has no CRS, skipping: {shp_path}")
            return None

        gdf = clean_geometries(gdf)
        print(f"Loaded optional {layer_name} boundary: {shp_path}")
        return gdf

    except Exception as e:
        print(f"Could not load optional {layer_name} boundary: {e}")
        return None


def add_study_outline(ax, soc, crs, coastal_boundary=None):
    try:
        soc_plot = soc.to_crs(crs)
        soc_plot.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.8,
            zorder=4
        )
    except Exception:
        pass

    try:
        if coastal_boundary is not None:
            coastal_plot = coastal_boundary.to_crs(crs)
            coastal_plot.boundary.plot(
                ax=ax,
                color="black",
                linewidth=1.2,
                linestyle="--",
                zorder=5
            )
    except Exception:
        pass


def add_city_labels(ax, crs):
    """
    Adds city markers and labels to individual maps only.
    City coordinates are stored in lon/lat, EPSG:4326, and reprojected to the map CRS.
    """
    try:
        city_df = pd.DataFrame(
            [
                {"city": city, "lon": lon, "lat": lat}
                for city, (lon, lat) in CITIES.items()
            ]
        )

        city_gdf = gpd.GeoDataFrame(
            city_df,
            geometry=gpd.points_from_xy(city_df["lon"], city_df["lat"]),
            crs="EPSG:4326"
        )

        city_gdf = city_gdf.to_crs(crs)

        for _, row in city_gdf.iterrows():
            x = row.geometry.x
            y = row.geometry.y

            ax.scatter(
                x,
                y,
                s=18,
                marker="o",
                color="black",
                edgecolor="white",
                linewidth=0.4,
                zorder=7
            )

            ax.annotate(
                row["city"],
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="black",
                path_effects=None,
                zorder=8
            )

    except Exception as e:
        print(f"Could not add city labels: {e}")


def save_single_correlation_map(
    corr_da,
    soc,
    out_png,
    title,
    smooth_display=False,
    coastal_boundary=None,
    add_cities=True
):
    plot_da = get_display_da(
        corr_da,
        "correlation",
        smooth_display=smooth_display
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    im = plot_da.plot(
        ax=ax,
        cmap=CORR_CMAP,
        vmin=-1,
        vmax=1,
        add_colorbar=False
    )

    add_study_outline(ax, soc, plot_da.rio.crs,
                      coastal_boundary=coastal_boundary)

    if add_cities:
        add_city_labels(ax, plot_da.rio.crs)

    if smooth_display:
        title = title + "\n3x3 smoothed display only"

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Spearman R")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved correlation map: {out_png}")


def save_single_lag_map(
    lag_da,
    soc,
    out_png,
    title,
    smooth_display=False,
    coastal_boundary=None,
    add_cities=True
):
    plot_da = get_display_da(
        lag_da,
        "lag",
        smooth_display=smooth_display
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    im = plot_da.plot(
        ax=ax,
        cmap=LAG_CMAP,
        norm=LAG_NORM,
        add_colorbar=False
    )

    add_study_outline(ax, soc, plot_da.rio.crs,
                      coastal_boundary=coastal_boundary)

    if add_cities:
        add_city_labels(ax, plot_da.rio.crs)

    if smooth_display:
        title = title + "\n3x3 majority-filter display only"

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        shrink=0.8,
        ticks=LAGS
    )

    cbar.set_label("PPT-leading response lag, months")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved lag map: {out_png}")


def ensure_all_variables_in_map_dict(map_dict, template_da, name_suffix):
    out = {}

    for var in PLOT_VARIABLE_ORDER:
        if var in map_dict:
            out[var] = map_dict[var]
        else:
            out[var] = make_blank_da_like(
                template_da,
                name=f"{var}_{name_suffix}_missing_or_no_output"
            )

    return out


def save_gridded_maps(
    map_dict,
    soc,
    out_png,
    title,
    map_type,
    template_da,
    smooth_display=False,
    coastal_boundary=None
):
    complete_map_dict = ensure_all_variables_in_map_dict(
        map_dict=map_dict,
        template_da=template_da,
        name_suffix=map_type
    )

    variables = PLOT_VARIABLE_ORDER

    ncols = 3
    nrows = int(np.ceil(len(variables) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 4.2 * nrows)
    )

    axes = np.atleast_1d(axes).ravel()

    last_im = None

    for ax, var_name in zip(axes, variables):
        da = complete_map_dict[var_name]
        plot_da = get_display_da(
            da,
            map_type,
            smooth_display=smooth_display
        )

        if map_type == "correlation":
            last_im = plot_da.plot(
                ax=ax,
                cmap=CORR_CMAP,
                vmin=-1,
                vmax=1,
                add_colorbar=False
            )

        elif map_type == "lag":
            last_im = plot_da.plot(
                ax=ax,
                cmap=LAG_CMAP,
                norm=LAG_NORM,
                add_colorbar=False
            )

        else:
            raise ValueError(f"Unknown map_type: {map_type}")

        add_study_outline(ax, soc, plot_da.rio.crs,
                          coastal_boundary=coastal_boundary)

        ax.set_title(var_name, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    for ax in axes[len(variables):]:
        ax.set_axis_off()

    if last_im is not None:
        if map_type == "correlation":
            cbar = fig.colorbar(
                last_im,
                ax=axes.tolist(),
                orientation="horizontal",
                fraction=0.04,
                pad=0.05
            )
            cbar.set_label("Spearman R")

        else:
            cbar = fig.colorbar(
                last_im,
                ax=axes.tolist(),
                orientation="horizontal",
                fraction=0.04,
                pad=0.05,
                ticks=LAGS
            )
            cbar.set_label("PPT-leading response lag, months")

    if smooth_display:
        title = title + "\n3x3 smoothed display only"

    fig.suptitle(title, fontsize=15, y=0.98)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved gridded map: {out_png}")


def save_rasters(map_result_by_mode, out_dir, var_name):
    if not SAVE_RASTERS:
        return

    raster_dir = Path(out_dir) / "GeoTIFFs"
    raster_dir.mkdir(parents=True, exist_ok=True)

    for mode_key, mode_result in map_result_by_mode.items():
        for map_kind, da in mode_result.items():
            da.rio.to_raster(
                raster_dir /
                f"{safe_name(var_name)}_{safe_name(mode_key)}_{map_kind}.tif"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading static data...")

    Path(BASE_OUT).mkdir(parents=True, exist_ok=True)

    soc = gpd.read_file(SOCAL_SHP)
    coastal_boundary = load_optional_boundary(
        COASTAL_ZONE_SHP, "California coastal zone")

    ndvi_base = load_ndvi_base(GIMMS_PATH, soc)

    template_2d = ndvi_base.isel(time=0)
    template_2d = ensure_spatial_dims(template_2d)

    static_layers = prepare_static_habitat_layers(template_2d)

    print("Loading whiplash events for three detection methods...")

    for detection_method, k_files in WHIPLASH_EVENT_FILES.items():
        print("\n" + "#" * 79)
        print(f"WHIPLASH DETECTION METHOD: {detection_method}")
        print("#" * 79)
        print("Loading whiplash events...")

        event_frames = []

        for k_value in K_LIST:
            event_path = k_files[k_value]

            if not event_path.exists():
                print(f"Missing event file: {event_path}")
                continue

            print(f"Using K{k_value} event file: {event_path}")
            event_df = pd.read_csv(event_path)

            required_columns = {"StartDate", "EndDate", "Direction"}
            missing_columns = sorted(required_columns - set(event_df.columns))
            if missing_columns:
                raise ValueError(
                    f"Whiplash event file is missing required columns {missing_columns}: "
                    f"{event_path}"
                )

            event_df["StartDate"] = pd.to_datetime(event_df["StartDate"])
            event_df["EndDate"] = pd.to_datetime(event_df["EndDate"])

            # Preserve the K column expected by the unchanged lag analysis.
            event_df["K"] = int(k_value)
            event_frames.append(event_df)

        if not event_frames:
            raise FileNotFoundError(
                f"No whiplash event files were loaded for {detection_method}."
            )

        df_all = pd.concat(event_frames, ignore_index=True)
        df_all = df_all.loc[df_all["K"].isin(K_LIST)].copy()

        if df_all.empty:
            raise ValueError(
                f"The loaded files contained no K3, K6, K12, or K24 events "
                f"for {detection_method}."
            )

        print(f"Loaded {len(df_all):,} events for {detection_method}.")
        print(df_all["K"].value_counts().sort_index().to_string())

        for direction in DIRECTIONS:
            print("\n==================================================")
            print(f"Direction: {direction}")
            print("==================================================")

            if direction == "Combined":
                df_dir = df_all.copy()
            else:
                df_dir = df_all.loc[
                    df_all["Direction"] == direction
                ].copy()

            if df_dir.empty:
                print(f"No events found for {direction}. Skipping.")
                continue

            base_event_months = get_event_months(df_dir)

            if len(base_event_months) < MIN_SAMPLES:
                print(
                    f"Not enough event months in analysis period for {direction}. Skipping.")
                continue

            print(
                f"Event-window months used for correlation: {len(base_event_months)}")

            for win_label, win_len in WINDOWS.items():
                print(f"\nWindow: {win_label}")

                months_load = get_full_analysis_load_months(
                    win_len=win_len,
                    max_lag=MAX_LAG
                )

                out_dir = (
                    Path(BASE_OUT)
                    / safe_name(detection_method)
                    / "ALL_Ks"
                    / safe_name(direction)
                    / safe_name(win_label)
                )

                out_dir.mkdir(parents=True, exist_ok=True)

                grid_dir = out_dir / "Gridded_Maps"
                csv_dir = out_dir / "CSV_Summaries"
                habitat_plot_dir = out_dir / "Habitat_Barplots"

                grid_dir.mkdir(parents=True, exist_ok=True)
                csv_dir.mkdir(parents=True, exist_ok=True)
                habitat_plot_dir.mkdir(parents=True, exist_ok=True)

                # Individual map folders by mode.
                individual_dirs = {}

                for mode_key in MAP_MODES.keys():
                    individual_dirs[mode_key] = {
                        "corr": out_dir / f"Individual_Correlation_Maps_{mode_key}",
                        "lag": out_dir / f"Individual_Lag_Maps_{mode_key}",
                    }

                    individual_dirs[mode_key]["corr"].mkdir(
                        parents=True, exist_ok=True)
                    individual_dirs[mode_key]["lag"].mkdir(
                        parents=True, exist_ok=True)

                    if SAVE_SMOOTHED_VISUAL_MAPS and mode_key in SMOOTHED_MODE_KEYS:
                        individual_dirs[mode_key]["corr_smooth"] = (
                            out_dir /
                            f"Individual_Correlation_Maps_{mode_key}_SmoothedDisplay"
                        )
                        individual_dirs[mode_key]["lag_smooth"] = (
                            out_dir /
                            f"Individual_Lag_Maps_{mode_key}_SmoothedDisplay"
                        )

                        individual_dirs[mode_key]["corr_smooth"].mkdir(
                            parents=True, exist_ok=True)
                        individual_dirs[mode_key]["lag_smooth"].mkdir(
                            parents=True, exist_ok=True)

                print("Loading PPT...")

                ppt_cube = load_prism_cube(
                    PPT_DIR,
                    template_2d,
                    months_load,
                    win_len,
                    var_name="PPT"
                )

                # Nested dictionaries:
                # map_collections[mode_key]["corr"][var_name] = da
                # map_collections[mode_key]["lag"][var_name] = da
                map_collections = {}

                for mode_key in MAP_MODES.keys():
                    map_collections[mode_key] = {
                        "corr": {},
                        "lag": {},
                    }

                all_habitat_corr = []
                map_summary_records = []
                sign_diagnostic_records = []
                correlation_bin_records = []
                lag_distribution_records = []
                failure_records = []

                for var_name, source in DRIVER_SPECS:
                    print(
                        f"\nRunning PPT-leading lagged correlation: PPT vs {var_name}")

                    try:
                        if source == "ndvi":
                            response_cube = load_ndvi_driver(
                                ndvi_base,
                                months_load,
                                win_len
                            )

                        elif source == "smrz":
                            response_cube = load_smrz_cube(
                                SMRZ_DIR,
                                template_2d,
                                soc,
                                months_load,
                                win_len
                            )

                        else:
                            response_cube = load_prism_cube(
                                source,
                                template_2d,
                                months_load,
                                win_len,
                                var_name=var_name
                            )

                        map_result_by_mode = compute_best_lag_maps(
                            ppt_cube=ppt_cube,
                            response_cube=response_cube,
                            base_months=base_event_months,
                            variable=var_name
                        )

                        save_rasters(
                            map_result_by_mode=map_result_by_mode,
                            out_dir=out_dir,
                            var_name=var_name
                        )

                        map_summary = summarize_map_outputs(
                            map_result_by_mode=map_result_by_mode,
                            var_name=var_name
                        )

                        map_summary_records.append(map_summary)

                        sign_df, bin_df, lag_df = make_all_diagnostic_tables(
                            map_result_by_mode=map_result_by_mode,
                            variable=var_name,
                            direction=direction,
                            win_label=win_label
                        )

                        if not sign_df.empty:
                            sign_diagnostic_records.append(sign_df)

                        if not bin_df.empty:
                            correlation_bin_records.append(bin_df)

                        if not lag_df.empty:
                            lag_distribution_records.append(lag_df)

                        # Store maps for gridded figures.
                        for mode_key in MAP_MODES.keys():
                            map_collections[mode_key]["corr"][var_name] = (
                                map_result_by_mode[mode_key]["corr"]
                            )
                            map_collections[mode_key]["lag"][var_name] = (
                                map_result_by_mode[mode_key]["lag"]
                            )

                        if SAVE_INDIVIDUAL_MAPS:
                            for mode_key, mode_spec in MAP_MODES.items():
                                mode_label = mode_spec["label"]

                                save_single_correlation_map(
                                    corr_da=map_result_by_mode[mode_key]["corr"],
                                    soc=soc,
                                    out_png=individual_dirs[mode_key]["corr"] /
                                    f"PPT_vs_{safe_name(var_name)}_{mode_key}_correlation.png",
                                    title=(
                                        f"PPT vs {var_name}: best correlation\n"
                                        f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                    ),
                                    smooth_display=False,
                                    coastal_boundary=coastal_boundary
                                )

                                save_single_lag_map(
                                    lag_da=map_result_by_mode[mode_key]["lag"],
                                    soc=soc,
                                    out_png=individual_dirs[mode_key]["lag"] /
                                    f"PPT_vs_{safe_name(var_name)}_{mode_key}_lag.png",
                                    title=(
                                        f"PPT vs {var_name}: best response lag\n"
                                        f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                    ),
                                    smooth_display=False,
                                    coastal_boundary=coastal_boundary
                                )

                                if SAVE_SMOOTHED_VISUAL_MAPS and mode_key in SMOOTHED_MODE_KEYS:
                                    save_single_correlation_map(
                                        corr_da=map_result_by_mode[mode_key]["corr"],
                                        soc=soc,
                                        out_png=individual_dirs[mode_key]["corr_smooth"] /
                                        f"PPT_vs_{safe_name(var_name)}_{mode_key}_correlation_SMOOTHED_DISPLAY.png",
                                        title=(
                                            f"PPT vs {var_name}: best correlation\n"
                                            f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                        ),
                                        smooth_display=True,
                                        coastal_boundary=coastal_boundary
                                    )

                                    save_single_lag_map(
                                        lag_da=map_result_by_mode[mode_key]["lag"],
                                        soc=soc,
                                        out_png=individual_dirs[mode_key]["lag_smooth"] /
                                        f"PPT_vs_{safe_name(var_name)}_{mode_key}_lag_SMOOTHED_DISPLAY.png",
                                        title=(
                                            f"PPT vs {var_name}: best response lag\n"
                                            f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                        ),
                                        smooth_display=True,
                                        coastal_boundary=coastal_boundary
                                    )

                        habitat_corr = compute_habitat_lag_correlations(
                            ppt_cube=ppt_cube,
                            response_cube=response_cube,
                            static_layers=static_layers,
                            base_months=base_event_months,
                            var_name=var_name,
                            direction=direction,
                            win_label=win_label
                        )

                        if not habitat_corr.empty:
                            all_habitat_corr.append(habitat_corr)

                        del response_cube
                        gc.collect()

                    except Exception as e:
                        print(f"Failed PPT vs {var_name}: {e}")

                        failure_records.append({
                            "direction": direction,
                            "window": win_label,
                            "variable": var_name,
                            "error": str(e)
                        })

                        # Force blank maps so all seven variables appear in gridded outputs.
                        for mode_key in MAP_MODES.keys():
                            blank_corr = make_blank_da_like(
                                template_2d,
                                f"{var_name}_{mode_key}_corr_missing"
                            )
                            blank_lag = make_blank_da_like(
                                template_2d,
                                f"{var_name}_{mode_key}_lag_missing"
                            )

                            map_collections[mode_key]["corr"][var_name] = blank_corr
                            map_collections[mode_key]["lag"][var_name] = blank_lag

                        # Failure row for diagnostics.
                        fail_rows = []

                        for mode_key, mode_spec in MAP_MODES.items():
                            fail_rows.append({
                                "variable": var_name,
                                "mode": mode_key,
                                "mode_label": mode_spec["label"],
                                "expected_sign": EXPECTED_SIGN_BY_VARIABLE.get(var_name, "absolute"),
                                "mapped_pixels": 0,
                                "min_best_r": np.nan,
                                "max_best_r": np.nan,
                                "median_best_r": np.nan,
                                "min_best_p": np.nan,
                                "median_best_p": np.nan,
                                "alpha": ALPHA,
                                "r_threshold": mode_spec["r_threshold"],
                                "uses_expected_sign": mode_spec["expected_sign"],
                                "uses_p_filter": mode_spec["p_filter"],
                                "error": str(e),
                            })

                        map_summary_records.append(pd.DataFrame(fail_rows))
                        gc.collect()

                # Save diagnostics.
                if len(map_summary_records) > 0:
                    map_summary_all = pd.concat(
                        map_summary_records,
                        ignore_index=True
                    )

                    summary_csv = (
                        csv_dir
                        / f"Map_Diagnostics_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    map_summary_all.to_csv(summary_csv, index=False)
                    print(f"Saved map diagnostics: {summary_csv}")

                sign_diagnostics_all = pd.DataFrame()
                correlation_bins_all = pd.DataFrame()
                lag_distribution_all = pd.DataFrame()

                if len(sign_diagnostic_records) > 0:
                    sign_diagnostics_all = pd.concat(
                        sign_diagnostic_records,
                        ignore_index=True
                    )

                    sign_csv = (
                        csv_dir
                        / f"Expected_vs_Opposite_Sign_Diagnostics_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    sign_diagnostics_all.to_csv(sign_csv, index=False)
                    print(
                        f"Saved expected/opposite sign diagnostics: {sign_csv}")

                if len(correlation_bin_records) > 0:
                    correlation_bins_all = pd.concat(
                        correlation_bin_records,
                        ignore_index=True
                    )

                    bin_csv = (
                        csv_dir
                        / f"Correlation_Bin_Diagnostics_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    correlation_bins_all.to_csv(bin_csv, index=False)
                    print(f"Saved correlation-bin diagnostics: {bin_csv}")

                if len(lag_distribution_records) > 0:
                    lag_distribution_all = pd.concat(
                        lag_distribution_records,
                        ignore_index=True
                    )

                    lag_csv = (
                        csv_dir
                        / f"Lag_Distribution_Percentages_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    lag_distribution_all.to_csv(lag_csv, index=False)
                    print(f"Saved lag distribution percentages: {lag_csv}")

                if SAVE_DIAGNOSTIC_BAR_PLOTS:
                    save_diagnostic_barplots(
                        sign_df=sign_diagnostics_all,
                        bin_df=correlation_bins_all,
                        lag_df=lag_distribution_all,
                        out_dir=out_dir,
                        direction=direction,
                        win_label=win_label,
                    )

                if len(failure_records) > 0:
                    failure_csv = (
                        csv_dir
                        / f"Variable_Loading_Failures_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    pd.DataFrame(failure_records).to_csv(
                        failure_csv, index=False)
                    print(f"Saved failure diagnostics: {failure_csv}")

                # Save gridded maps for every mode.
                if SAVE_GRIDDED_MAPS:
                    for mode_key, mode_spec in MAP_MODES.items():
                        mode_label = mode_spec["label"]

                        save_gridded_maps(
                            map_dict=map_collections[mode_key]["corr"],
                            soc=soc,
                            out_png=grid_dir /
                            f"Gridded_Correlation_Maps_{mode_key}_{safe_name(direction)}_{safe_name(win_label)}.png",
                            title=(
                                f"Best Spearman correlation with PPT\n"
                                f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                            ),
                            map_type="correlation",
                            template_da=template_2d,
                            smooth_display=False,
                            coastal_boundary=coastal_boundary
                        )

                        save_gridded_maps(
                            map_dict=map_collections[mode_key]["lag"],
                            soc=soc,
                            out_png=grid_dir /
                            f"Gridded_Lag_Maps_{mode_key}_{safe_name(direction)}_{safe_name(win_label)}.png",
                            title=(
                                f"Best PPT-leading response lag\n"
                                f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                            ),
                            map_type="lag",
                            template_da=template_2d,
                            smooth_display=False,
                            coastal_boundary=coastal_boundary
                        )

                        if SAVE_SMOOTHED_VISUAL_MAPS and mode_key in SMOOTHED_MODE_KEYS:
                            save_gridded_maps(
                                map_dict=map_collections[mode_key]["corr"],
                                soc=soc,
                                out_png=grid_dir /
                                f"Gridded_Correlation_Maps_{mode_key}_SMOOTHED_DISPLAY_{safe_name(direction)}_{safe_name(win_label)}.png",
                                title=(
                                    f"Best Spearman correlation with PPT\n"
                                    f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                ),
                                map_type="correlation",
                                template_da=template_2d,
                                smooth_display=True,
                                coastal_boundary=coastal_boundary
                            )

                            save_gridded_maps(
                                map_dict=map_collections[mode_key]["lag"],
                                soc=soc,
                                out_png=grid_dir /
                                f"Gridded_Lag_Maps_{mode_key}_SMOOTHED_DISPLAY_{safe_name(direction)}_{safe_name(win_label)}.png",
                                title=(
                                    f"Best PPT-leading response lag\n"
                                    f"{direction}, {win_label}, {CLIMATE_TRANSFORM}, {mode_label}"
                                ),
                                map_type="lag",
                                template_da=template_2d,
                                smooth_display=True,
                                coastal_boundary=coastal_boundary
                            )

                # Save habitat correlations and habitat bar plot.
                if len(all_habitat_corr) > 0:
                    habitat_corr_all = pd.concat(
                        all_habitat_corr,
                        ignore_index=True
                    )

                    habitat_corr_csv = (
                        csv_dir
                        / f"Habitat_Lagged_Correlations_{safe_name(direction)}_{safe_name(win_label)}.csv"
                    )

                    habitat_corr_all.to_csv(habitat_corr_csv, index=False)
                    print(f"Saved habitat correlation CSV: {habitat_corr_csv}")

                    if SAVE_HABITAT_BAR_PLOTS:
                        save_habitat_lag_barplots(
                            hab_corr_df=habitat_corr_all,
                            out_dir=out_dir,
                            direction=direction,
                            win_label=win_label
                        )

                del ppt_cube
                gc.collect()

    print("DONE - ALL THREE DETECTION METHODS")


if __name__ == "__main__":
    main()
