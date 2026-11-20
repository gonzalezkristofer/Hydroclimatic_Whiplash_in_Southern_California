"""
Standalone combined NDVI and SMrz whiplash-response analysis.

Runs both variables for:
- Original_MultiK
- Chronological_First_Opposite
- Shortest_First

Uses the exact SD_0.8 K-specific event files and restricts all response data
and event dates to January 1982 through December 2022.
"""

import json
import os
from pathlib import Path
import glob
import re
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import contextily as cx
from scipy.stats import wilcoxon, kruskal
from dateutil.relativedelta import relativedelta
from rasterio.transform import xy
from pyproj import Geod
import statsmodels.formula.api as smf
import statsmodels.api as sm
os.environ.setdefault(
    'GDAL_DATA', 'C:\\Users\\Owner\\anaconda3\\envs\\myenv\\Library\\share\\gdal')
os.environ.setdefault(
    'PROJ_LIB', 'C:\\Users\\Owner\\anaconda3\\envs\\myenv\\Library\\share\\proj')
ndvi_FINAL_PAPER_ROOT = Path('D:\\Graduate Research\\Final_Paper_Analysis')
ndvi_BASE_OUT_ROOT = ndvi_FINAL_PAPER_ROOT / \
    'Whiplash_Response_NDVI_SMRz_1982_2022'
ndvi_WHIPLASH_ROOT = ndvi_FINAL_PAPER_ROOT / \
    'Whiplash Analysis Precipitation 1982_2022'
ndvi_GIMMS_DIR = 'D:\\Graduate Research\\Graduate Research\\Data\\GIMMS NDVI Data\\GIMMS NDVI Data.nc'
ndvi_SOCAL_SHP = 'D:\\Graduate Research\\Graduate Research\\Data\\BoundingBox\\BoundingBox.shp'
COUNTY_SHP = 'D:\\Graduate Research\\Graduate Research\\Study Area Extent\\ca_counties\\CA_Counties.shp'
ndvi_VEG_SHPS = {'Agriculture': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Agriculturen_4km.shp', 'Conifer_Forest': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Conifer_Forest_4km.shp', 'Conifer_Woodland': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Conifer_Woodland_4km.shp', 'Desert': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Desert_4km.shp',
                 'Hardwood': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Hardwood_4km.shp', 'Herbaceous': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Herbaceous_4km.shp', 'Shrub': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Shrub_4km.shp', 'Urban': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Urban_4km.shp'}
ndvi_ELEV_SHP = 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Habitats_Elevation\\Elevation_Different_Resolutions\\elevation_classified_8km_250m_bins_land_northup.shp'
ndvi_ELEV_SHP_FALLBACK = 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Habitats_Elevation\\Elevation_Different_Resolutions\\elevation_classified_1km_250m_bins_land_northup.shp'
ndvi_VARIABLE_NAME = 'NDVI'
ndvi_METRIC = 'MEAN'
ndvi_K_LIST = [3, 6, 12, 24]
ndvi_TARGET_WINDOW = '3-Month Mean'
ndvi_DIRECTIONS = ['Dry->Wet', 'Wet->Dry']
ndvi_BUFFER_MONTHS = 3
ndvi_MIN_EVENTS = 6
ndvi_ALPHA = 0.05
ndvi_ELEV_BIN_FIELD = None
ndvi_ELEV_MIDPOINT_FIELD = None
ndvi_ELEV_BIN_SIZE_M = 250
ndvi_SAVE_RASTERS = True
ndvi_SAVE_STATS = True
ndvi_HABITAT_ORDER = list(ndvi_VEG_SHPS.keys())
ndvi_REFERENCE_HABITAT = 'Shrub'
ndvi_REFERENCE_TRANSITION = 'Dry_to_Wet'
ndvi_REFERENCE_ELEV_BIN = '0-250 m'
ndvi_ANALYSIS_START = pd.Timestamp('1982-01-01')
ndvi_ANALYSIS_END = pd.Timestamp('2022-12-31')
ndvi_DETECTION_METHODS = ['Original_MultiK',
                          'Chronological_First_Opposite', 'Shortest_First']
ndvi_K_FILE_LABELS = {3: '1-3 months', 6: '4-6 months',
                      12: '7-12 months', 24: '13-24 months'}


def ndvi_get_whiplash_event_path(detection_method, k_value):
    """Return the exact K-specific SD_0.8 CSV created by the detection script."""
    if detection_method not in ndvi_DETECTION_METHODS:
        raise ValueError(
            f'Unknown detection method: {detection_method}. Expected one of: {ndvi_DETECTION_METHODS}')
    if k_value not in ndvi_K_FILE_LABELS:
        raise ValueError(f'Unsupported K value: {k_value}')
    filename = f'Whiplash_Events_{detection_method}_SD_0.8_K{k_value}_{ndvi_K_FILE_LABELS[k_value]}.csv'
    return ndvi_WHIPLASH_ROOT / detection_method / 'SD_0.8' / filename


def ndvi_load_whiplash_events(detection_method):
    """
    Load all four exact K-specific event files for one detection method.

    Files are required to exist, including an empty K24 file where applicable.
    Events are restricted to the 3-Month Mean precipitation series and to
    January 1982 through December 2022.
    """
    expected_paths = {k_value: ndvi_get_whiplash_event_path(
        detection_method, k_value) for k_value in ndvi_K_LIST}
    missing_paths = [path for path in expected_paths.values()
                     if not path.exists()]
    if missing_paths:
        missing_text = '\n'.join((f'  - {path}' for path in missing_paths))
        raise FileNotFoundError(
            f'Missing required whiplash event file(s) for {detection_method}:\n{missing_text}')
    all_k_frames = []
    for k_value in ndvi_K_LIST:
        df_path = expected_paths[k_value]
        print(f'Using K{k_value} event file: {df_path}')
        df_all = pd.read_csv(df_path)
        required_columns = {'StartDate', 'EndDate', 'Direction'}
        missing_columns = sorted(required_columns - set(df_all.columns))
        if missing_columns:
            raise ValueError(
                f'Event file is missing required columns {missing_columns}: {df_path}')
        df_all['StartDate'] = pd.to_datetime(
            df_all['StartDate'], errors='coerce')
        df_all['EndDate'] = pd.to_datetime(df_all['EndDate'], errors='coerce')
        df_all = df_all.dropna(
            subset=['StartDate', 'EndDate', 'Direction']).copy()
        if 'Window' in df_all.columns:
            df_all = df_all.loc[df_all['Window'].eq(ndvi_TARGET_WINDOW)].copy()
        df_all = df_all.loc[df_all['StartDate'].ge(
            ndvi_ANALYSIS_START) & df_all['EndDate'].le(ndvi_ANALYSIS_END)].copy()
        df_all['K'] = int(k_value)
        if not df_all.empty:
            all_k_frames.append(df_all)
        print(
            f'  Loaded {len(df_all):,} usable K{k_value} events for {detection_method}.')
    if not all_k_frames:
        raise ValueError(
            f'No usable January 1982-December 2022 events remained for {detection_method} and Window={ndvi_TARGET_WINDOW}.')
    df_all_k = pd.concat(all_k_frames, ignore_index=True)
    df_all_k = df_all_k.sort_values(
        ['StartDate', 'EndDate', 'K', 'Direction']).reset_index(drop=True)
    print(f'\nCombined All K event counts for {detection_method}:')
    print(df_all_k['Direction'].value_counts().to_string())
    print('\nCounts by K:')
    print(df_all_k['K'].value_counts().sort_index().to_string())
    return df_all_k


ndvi_RESPONSE_COLORS = {'Strong decrease (< -1 SD)': '#d7191c', 'Moderate decrease (-1 SD to 0)': '#fdae61', 'Not significant': '#d9d9d9',
                        'Zero significant change': '#bdbdbd', 'Moderate increase (0 to +1 SD)': '#a6d96a', 'Strong increase (> +1 SD)': '#1a9641'}
ndvi_RESPONSE_CLASS_ORDER = ['Strong decrease (< -1 SD)', 'Moderate decrease (-1 SD to 0)',
                             'Not significant', 'Moderate increase (0 to +1 SD)', 'Strong increase (> +1 SD)']
ndvi_DIRECTION_COLORS = {
    'Any significant decrease': '#d7191c', 'Any significant increase': '#1a9641'}
ndvi_cmap_5 = ListedColormap(
    ['#d7191c', '#fdae61', '#d9d9d9', '#a6d96a', '#1a9641'])
ndvi_RESPONSE_LABELS = {0: 'Strong decrease (< -1 SD)', 1: 'Moderate decrease (-1 SD to 0)', 2: 'Zero significant change',
                        3: 'Moderate increase (0 to +1 SD)', 4: 'Strong increase (> +1 SD)', 9: 'Not significant'}
ndvi_SIGNIFICANCE_STARS = {'Dry_to_Wet': {'Conifer_Forest': '***'}, 'Wet_to_Dry': {
    'Agriculture': '*', 'Conifer_Forest': '***', 'Herbaceous': '***'}}
ndvi_SIGNIFICANCE_NOTE = 'Asterisks indicate significant event-level model terms relative to Shrub: * p < 0.05, ** p < 0.01, *** p < 0.001. Dry->Wet markers show habitat main effects; Wet->Dry markers show transition x habitat interactions.'
ndvi_CITIES = {'Los Angeles': (-118.2437, 34.0522), 'San Diego': (-117.1611, 32.7157), 'Santa Barbara': (-119.6982,
                                                                                                         34.4208), 'Palm Springs': (-116.5453, 33.8303), 'Lancaster': (-118.1542, 34.6868), 'Oxnard': (-119.1771, 34.1975)}


def ndvi_expand_months(date):
    s = date - relativedelta(months=ndvi_BUFFER_MONTHS)
    e = date + relativedelta(months=ndvi_BUFFER_MONTHS)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime('%Y-%m'))
        cur += relativedelta(months=1)
    return out


def ndvi_run_wilcoxon(x):
    x = x[~np.isnan(x)]
    if len(x) < ndvi_MIN_EVENTS or np.all(x == 0):
        return (np.nan, 1.0)
    try:
        _, p = wilcoxon(x)
        return (np.nanmean(x), p)
    except Exception:
        return (np.nanmean(x), 1.0)


def ndvi_open_gimms_dataset(gimms_path):
    if os.path.isfile(gimms_path):
        return xr.open_dataset(gimms_path)
    if os.path.isdir(gimms_path):
        nc_files = sorted(glob.glob(os.path.join(gimms_path, '*.nc*')))
        if len(nc_files) == 0:
            raise FileNotFoundError(
                f'No .nc files found inside directory: {gimms_path}')
        return xr.open_mfdataset(nc_files, combine='by_coords', data_vars='all', coords='minimal', compat='override')
    try:
        return xr.open_dataset(gimms_path)
    except Exception as first_error:
        parent_dir = os.path.dirname(gimms_path)
        if os.path.isdir(parent_dir):
            nc_files = sorted(glob.glob(os.path.join(parent_dir, '*.nc*')))
            if len(nc_files) > 0:
                return xr.open_mfdataset(nc_files, combine='by_coords', data_vars='all', coords='minimal', compat='override')
        raise RuntimeError(
            f'Could not open GIMMS NDVI source.\nProvided path: {gimms_path}\nOriginal error: {first_error}')


def ndvi_clean_geometries(gdf):
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    try:
        gdf['geometry'] = gdf.geometry.make_valid()
    except Exception:
        gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def ndvi_ensure_spatial_dims(da):
    possible_x = ['x', 'lon', 'longitude']
    possible_y = ['y', 'lat', 'latitude']
    x_dim = None
    y_dim = None
    for d in da.dims:
        if d.lower() in possible_x:
            x_dim = d
        if d.lower() in possible_y:
            y_dim = d
    if x_dim is not None and y_dim is not None:
        da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326', inplace=False)
    return da


def ndvi_weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def ndvi_parse_midpoint_from_bin_label(label):
    if pd.isna(label):
        return np.nan
    s = str(label).strip().replace(',', '')
    s = s.replace('–', '-').replace('—', '-')
    s = re.sub('\\s+', ' ', s)
    if re.fullmatch('\\d+', s):
        code = int(s)
        low = code * ndvi_ELEV_BIN_SIZE_M
        high = (code + 1) * ndvi_ELEV_BIN_SIZE_M
        return (low + high) / 2.0
    nums = re.findall('\\d+\\.?\\d*', s)
    if len(nums) >= 2:
        low = float(nums[0])
        high = float(nums[1])
        return (low + high) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def ndvi_normalize_elevation_bin_label(label):
    """Return clean positive elevation-bin labels such as 0-250 m.

    Handles class IDs like 0, 1, 2; labels like 0 m, 1 m, 2 m;
    and range labels such as 0-250 m. Minus signs are treated as range
    separators so accidental labels like -250-0 m become 0-250 m.
    """
    if pd.isna(label):
        return 'Unknown'
    s = str(label).strip().replace(',', '')
    s = s.replace('–', '-').replace('—', '-')
    s = re.sub(r'\s+', ' ', s)

    class_match = re.fullmatch(r'(-?\d+(?:\.0+)?)\s*m?', s)
    if class_match:
        val = float(class_match.group(1))
        if val.is_integer() and 0 <= int(val) <= 50:
            code = int(val)
            low = code * ndvi_ELEV_BIN_SIZE_M
            high = (code + 1) * ndvi_ELEV_BIN_SIZE_M
            return f'{low}-{high} m'

    nums = re.findall(r'\d+\.?\d*', s)
    if len(nums) >= 2:
        low = float(nums[0])
        high = float(nums[1])
        low, high = sorted([low, high])
        if low.is_integer():
            low = int(low)
        if high.is_integer():
            high = int(high)
        return f'{low}-{high} m'
    return s.replace('--', '-')


def ndvi_make_elevation_label(label):
    return ndvi_normalize_elevation_bin_label(label)


def ndvi_choose_elevation_bin_field(elev_gdf):
    if ndvi_ELEV_BIN_FIELD is not None:
        if ndvi_ELEV_BIN_FIELD not in elev_gdf.columns:
            raise ValueError(
                f"ELEV_BIN_FIELD '{ndvi_ELEV_BIN_FIELD}' was not found.\nAvailable fields: {list(elev_gdf.columns)}")
        return ndvi_ELEV_BIN_FIELD
    candidates = ['elev_bin', 'ELEV_BIN', 'Elevation_Bin', 'elevation_bin', 'class_id', 'CLASS_ID', 'class',
                  'Class', 'CLASS', 'gridcode', 'GRIDCODE', 'grid_code', 'GRID_CODE', 'bin', 'BIN', 'Value', 'VALUE', 'DN']
    for c in candidates:
        if c in elev_gdf.columns:
            return c
    for c in elev_gdf.columns:
        cl = c.lower()
        if 'elev' in cl or 'bin' in cl or 'class' in cl or ('grid' in cl):
            return c
    raise ValueError(
        f'Could not detect the elevation-bin field automatically.\nAvailable elevation fields are:\n{list(elev_gdf.columns)}\nSet ELEV_BIN_FIELD near the top of the script.')


def ndvi_get_unique_ordered_elevation_bins_from_static(static_df):
    tmp = static_df[['elev_bin', 'elev_mid_m']].dropna(
        subset=['elev_bin', 'elev_mid_m']).drop_duplicates().sort_values('elev_mid_m')
    out = []
    seen = set()
    for value in tmp['elev_bin'].tolist():
        label = ndvi_normalize_elevation_bin_label(value)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def ndvi_get_unique_ordered_elevation_bins_from_event_df(df):
    tmp = df[['elev_bin', 'mean_elev_m']].dropna(
        subset=['elev_bin', 'mean_elev_m']).drop_duplicates().sort_values('mean_elev_m')
    out = []
    seen = set()
    for value in tmp['elev_bin'].tolist():
        label = ndvi_normalize_elevation_bin_label(value)
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


# ============================================================
# COUNTY BASEMAP HELPERS FOR DELTA MAPS
# ============================================================
MAP_LAND_FILL_COLOR = '#C8C8C8'
MAP_OCEAN_BACKGROUND_COLOR = 'white'
MAP_COUNTY_EDGE_COLOR = '#4D4D4D'
MAP_COUNTY_LINEWIDTH = 0.45
MAP_COUNTY_ALPHA = 0.45
MAP_STUDY_BOUNDARY_COLOR = 'black'
MAP_STUDY_BOUNDARY_LINEWIDTH = 0.75
MAP_RASTER_ALPHA = 1.0
MAP_CLIP_RASTER_TO_COUNTY_LAND = True
INDIVIDUAL_MAP_FIGSIZE = (9.6, 5.8)


def map_clean_geometries(gdf):
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    try:
        gdf['geometry'] = gdf.geometry.make_valid()
    except Exception:
        gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def load_counties_clipped_to_study(soc):
    county_path = Path(COUNTY_SHP)
    if not county_path.exists():
        print(
            f'County shapefile not found, using study-area fill only: {county_path}')
        return None

    try:
        counties = gpd.read_file(county_path)
        if counties.crs is None:
            print('County shapefile has no CRS, using study-area fill only.')
            return None
        counties = map_clean_geometries(counties)
        counties = counties.to_crs(soc.crs)
        counties = gpd.clip(counties, soc)
        counties = map_clean_geometries(counties)
        return counties
    except Exception as e:
        print(f'Could not load county basemap layer: {e}')
        return None


def add_county_land_fill(ax, soc, target_crs):
    ax.set_facecolor(MAP_OCEAN_BACKGROUND_COLOR)

    counties = load_counties_clipped_to_study(soc)
    if counties is not None and not counties.empty:
        counties.to_crs(target_crs).plot(
            ax=ax,
            facecolor=MAP_LAND_FILL_COLOR,
            edgecolor='none',
            linewidth=0,
            alpha=1.0,
            zorder=1,
        )
    else:
        soc.to_crs(target_crs).plot(
            ax=ax,
            facecolor=MAP_LAND_FILL_COLOR,
            edgecolor='none',
            linewidth=0,
            alpha=1.0,
            zorder=1,
        )


def add_county_boundaries_and_study_outline(ax, soc, target_crs):
    counties = load_counties_clipped_to_study(soc)
    if counties is not None and not counties.empty:
        counties.to_crs(target_crs).boundary.plot(
            ax=ax,
            color=MAP_COUNTY_EDGE_COLOR,
            linewidth=MAP_COUNTY_LINEWIDTH,
            alpha=MAP_COUNTY_ALPHA,
            zorder=5,
        )

    soc.to_crs(target_crs).boundary.plot(
        ax=ax,
        color=MAP_STUDY_BOUNDARY_COLOR,
        linewidth=MAP_STUDY_BOUNDARY_LINEWIDTH,
        zorder=6,
        clip_on=False,
    )


def set_map_extent_to_study(ax, soc, target_crs, pad_x=0.0, pad_y=0.0):
    """Match the map frame to the study-domain extent, like the correlation and lag maps."""
    soc_plot = soc.to_crs(target_crs)
    xmin, ymin, xmax, ymax = soc_plot.total_bounds
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - dx * pad_x, xmax + dx * pad_x)
    ax.set_ylim(ymin - dy * pad_y, ymax + dy * pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0)


def plot_delta_class_raster(ax, da, cmap, vmin=0, vmax=4, zorder=3):
    """Plot classified delta pixels with imshow so panels keep the correct map shape."""
    plot_da = da.copy().squeeze(drop=True)

    data = np.asarray(plot_da.values, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f'Delta class raster must be 2-D after squeeze; got shape {data.shape}')

    masked_data = np.ma.masked_invalid(data)
    xmin, ymin, xmax, ymax = plot_da.rio.bounds()

    try:
        y_vals = plot_da[plot_da.rio.y_dim].values
        origin = 'upper' if y_vals[0] > y_vals[-1] else 'lower'
    except Exception:
        origin = 'upper'

    use_cmap = cmap.copy()
    use_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    im = ax.imshow(
        masked_data,
        extent=(xmin, xmax, ymin, ymax),
        origin=origin,
        cmap=use_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        zorder=zorder,
    )
    return im


def mask_delta_raster_to_county_land(da, soc):
    """Keep plotted delta pixels on the clipped county/study-domain land layer."""
    if not MAP_CLIP_RASTER_TO_COUNTY_LAND:
        return da

    try:
        counties = load_counties_clipped_to_study(soc)
        if counties is not None and not counties.empty:
            counties_plot = counties.to_crs(da.rio.crs)
            return da.rio.clip(
                counties_plot.geometry,
                counties_plot.crs,
                drop=False,
                all_touched=True,
            )
    except Exception as e:
        print(f'Could not mask delta raster to county land layer: {e}')

    try:
        soc_plot = soc.to_crs(da.rio.crs)
        return da.rio.clip(
            soc_plot.geometry,
            soc_plot.crs,
            drop=False,
            all_touched=True,
        )
    except Exception as e:
        print(f'Could not mask delta raster to study boundary: {e}')
        return da


def ndvi_add_city_labels(ax):
    city_df = pd.DataFrame([(name, lon, lat) for name, (lon, lat)
                           in ndvi_CITIES.items()], columns=['City', 'lon', 'lat'])
    city_gdf = gpd.GeoDataFrame(city_df, geometry=gpd.points_from_xy(
        city_df['lon'], city_df['lat']), crs='EPSG:4326').to_crs(epsg=3857)
    for _, row in city_gdf.iterrows():
        ax.plot(row.geometry.x, row.geometry.y, 'ko', markersize=3, zorder=5)
        ax.text(row.geometry.x + 6000, row.geometry.y + 6000, row['City'], fontsize=8, color='black', bbox=dict(
            facecolor='white', edgecolor='none', alpha=0.75, pad=1.2), zorder=6)


def ndvi_add_study_area_outline(ax, soc):
    soc_3857 = soc.to_crs(epsg=3857)
    soc_3857.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=4)


def ndvi_get_legend_for_sd_mode(meta):
    sd = meta['sd']
    return ([Patch(color='#1a9641', label=f'> +1 SD ({sd:.4f})'), Patch(color='#a6d96a', label='0 to +1 SD'), Patch(color='#d9d9d9', label='Not significant'), Patch(color='#fdae61', label='-1 SD to 0'), Patch(color='#d7191c', label=f'< -1 SD (-{sd:.4f})')], ndvi_cmap_5, 0, 4)


def ndvi_save_single_map(class_map, soc, out_file, title_text, meta):
    da = class_map.copy()
    da = ndvi_ensure_spatial_dims(da)

    # Keep the map in native lon/lat-style geometry rather than Web Mercator.
    # This matches the visual proportions of the correlation/lag figures and
    # prevents the delta maps from looking compressed into square panels.
    da_plot = mask_delta_raster_to_county_land(da, soc)

    legend_handles, use_cmap, vmin, vmax = ndvi_get_legend_for_sd_mode(meta)

    fig, ax = plt.subplots(1, 1, figsize=INDIVIDUAL_MAP_FIGSIZE)

    add_county_land_fill(ax, soc, da_plot.rio.crs)

    plot_delta_class_raster(
        ax=ax,
        da=da_plot,
        cmap=use_cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )

    set_map_extent_to_study(ax, soc, da_plot.rio.crs)
    add_county_boundaries_and_study_outline(ax, soc, da_plot.rio.crs)

    # No satellite basemap and no city labels; this matches the lag/correlation map style.
    ax.set_title(title_text, fontsize=14)
    ax.set_axis_off()

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.0, -0.08),
        ncol=2,
        fontsize=10,
    )

    plt.subplots_adjust(bottom=0.17, top=0.88)
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved map: {out_file}')


def ndvi_compute_event_deltas(ndvi, df_events):
    deltas = []
    for row_i, e in df_events.iterrows():
        s = ndvi.sel(time=ndvi['ym'].isin(ndvi_expand_months(e['StartDate'])))
        f = ndvi.sel(time=ndvi['ym'].isin(ndvi_expand_months(e['EndDate'])))
        if s.sizes.get('time', 0) == 0 or f.sizes.get('time', 0) == 0:
            continue
        d = f.mean('time') - s.mean('time')
        d = d.reset_coords(drop=True)
        event_id = f"K{int(e['K'])}_{str(e['Direction']).replace('->', 'to')}_{pd.to_datetime(e['StartDate']).strftime('%Y%m')}_{pd.to_datetime(e['EndDate']).strftime('%Y%m')}_{row_i}"
        deltas.append({'event_id': event_id, 'transition': str(e['Direction']).replace('->', '_to_'), 'K': int(
            e['K']), 'start_date': pd.to_datetime(e['StartDate']), 'end_date': pd.to_datetime(e['EndDate']), 'delta': d})
    return deltas


def ndvi_classify_sd_map(center_map, sig_mask, land_mask):
    vals = center_map.where(sig_mask & land_mask).values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (xr.full_like(center_map, np.nan), None)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return (xr.full_like(center_map, np.nan), None)
    out = xr.full_like(center_map, np.nan)
    out = out.where(~(center_map < -sd), 0)
    out = out.where(~((center_map >= -sd) & (center_map < 0)), 1)
    out = out.where(~(center_map == 0), 2)
    out = out.where(~((center_map > 0) & (center_map <= sd)), 3)
    out = out.where(~(center_map > sd), 4)
    out = out.where(sig_mask)
    out = out.where(land_mask)
    meta = {'sd': float(sd)}
    return (out, meta)


def ndvi_response_class_array(delta, sig_mask, sd):
    out = np.full(delta.shape, 9, dtype=np.int16)
    valid = np.isfinite(delta) & sig_mask
    out[valid & (delta < -sd)] = 0
    out[valid & (delta >= -sd) & (delta < 0)] = 1
    out[valid & (delta == 0)] = 2
    out[valid & (delta > 0) & (delta <= sd)] = 3
    out[valid & (delta > sd)] = 4
    return out


def ndvi_load_habitat_polygons():
    pieces = []
    for habitat, shp in ndvi_VEG_SHPS.items():
        shp_path = Path(shp)
        if not shp_path.exists():
            warnings.warn(f'Missing habitat shapefile: {shp_path}')
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError(f'Habitat shapefile has no CRS: {shp_path}')
        gdf = ndvi_clean_geometries(gdf)
        gdf['habitat'] = habitat
        pieces.append(gdf[['habitat', 'geometry']])
    if len(pieces) == 0:
        raise ValueError('No habitat shapefiles were loaded.')
    habitats = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True), crs=pieces[0].crs)
    habitats = ndvi_clean_geometries(habitats)
    return habitats


def ndvi_load_elevation_polygons():
    elev_path = Path(ndvi_ELEV_SHP)
    if not elev_path.exists():
        fallback = Path(ndvi_ELEV_SHP_FALLBACK)
        if not fallback.exists():
            raise FileNotFoundError(
                f'Elevation shapefile was not found:\n{ndvi_ELEV_SHP}\n{ndvi_ELEV_SHP_FALLBACK}')
        elev_path = fallback
        print(f'Using fallback elevation shapefile: {elev_path}')
    elev = gpd.read_file(elev_path)
    if elev.crs is None:
        raise ValueError(f'Elevation shapefile has no CRS: {elev_path}')
    elev = ndvi_clean_geometries(elev)
    bin_field = ndvi_choose_elevation_bin_field(elev)
    print(f'Using elevation-bin field: {bin_field}')
    elev['elev_bin_raw'] = elev[bin_field].astype(str)
    elev['elev_bin'] = elev['elev_bin_raw'].map(ndvi_make_elevation_label)
    if ndvi_ELEV_MIDPOINT_FIELD is not None:
        if ndvi_ELEV_MIDPOINT_FIELD not in elev.columns:
            raise ValueError(
                f"ELEV_MIDPOINT_FIELD '{ndvi_ELEV_MIDPOINT_FIELD}' was not found.\nAvailable fields: {list(elev.columns)}")
        elev['elev_mid_m'] = pd.to_numeric(
            elev[ndvi_ELEV_MIDPOINT_FIELD], errors='coerce')
    else:
        elev['elev_mid_m'] = elev['elev_bin_raw'].map(
            ndvi_parse_midpoint_from_bin_label)
    elev['elev_bin'] = elev['elev_bin'].map(ndvi_normalize_elevation_bin_label)
    return elev[['elev_bin', 'elev_mid_m', 'geometry']]


def ndvi_create_pixel_area_grid_km2(template_da):
    template_da = ndvi_ensure_spatial_dims(template_da)
    crs = template_da.rio.crs
    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape
    if crs is not None and crs.is_projected:
        pixel_area = abs(transform.a * transform.e) / 1000000.0
        return np.full((height, width), pixel_area, dtype=float)
    geod = Geod(ellps='WGS84')
    areas = np.full((height, width), np.nan, dtype=float)
    for r in range(height):
        for c in range(width):
            x_ul, y_ul = xy(transform, r, c, offset='ul')
            x_ur, y_ur = xy(transform, r, c, offset='ur')
            x_lr, y_lr = xy(transform, r, c, offset='lr')
            x_ll, y_ll = xy(transform, r, c, offset='ll')
            area_m2, _ = geod.polygon_area_perimeter(
                [x_ul, x_ur, x_lr, x_ll], [y_ul, y_ur, y_lr, y_ll])
            areas[r, c] = abs(area_m2) / 1000000.0
    return areas


def ndvi_make_pixel_center_table(template_da):
    template_da = ndvi_ensure_spatial_dims(template_da)
    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape
    records = []
    for r in range(height):
        for c in range(width):
            lon, lat = xy(transform, r, c, offset='center')
            records.append({'row': int(r), 'col': int(
                c), 'lon': float(lon), 'lat': float(lat)})
    pixels = pd.DataFrame.from_records(records)
    pixel_gdf = gpd.GeoDataFrame(pixels, geometry=gpd.points_from_xy(
        pixels['lon'], pixels['lat']), crs=template_da.rio.crs)
    return pixel_gdf


def ndvi_prepare_static_layers(template_da):
    print('\nPreparing static habitat and elevation layers using pixel-center joins...')
    template_da = ndvi_ensure_spatial_dims(template_da)
    habitat_gdf = ndvi_load_habitat_polygons()
    elev_gdf = ndvi_load_elevation_polygons()
    pixel_points = ndvi_make_pixel_center_table(template_da)
    habitat_join = gpd.sjoin(pixel_points, habitat_gdf.to_crs(
        pixel_points.crs), how='left', predicate='within')
    habitat_join = habitat_join.sort_values(
        ['row', 'col']).drop_duplicates(['row', 'col'])
    elev_join = gpd.sjoin(pixel_points, elev_gdf.to_crs(
        pixel_points.crs), how='left', predicate='within')
    elev_join = elev_join.sort_values(
        ['row', 'col', 'elev_mid_m']).drop_duplicates(['row', 'col'])
    static = pixel_points[['row', 'col', 'lon', 'lat', 'geometry']].copy()
    static = static.merge(habitat_join[['row', 'col', 'habitat']], on=[
                          'row', 'col'], how='left')
    static = static.merge(elev_join[['row', 'col', 'elev_bin', 'elev_mid_m']], on=[
                          'row', 'col'], how='left')
    static['elev_bin'] = static['elev_bin'].map(
        ndvi_normalize_elevation_bin_label)
    area_km2 = ndvi_create_pixel_area_grid_km2(template_da)
    area_records = []
    for r in range(area_km2.shape[0]):
        for c in range(area_km2.shape[1]):
            area_records.append({'row': int(r), 'col': int(
                c), 'area_km2': float(area_km2[r, c])})
    area_df = pd.DataFrame(area_records)
    static = static.merge(area_df, on=['row', 'col'], how='left')
    habitat_order = [
        h for h in ndvi_HABITAT_ORDER if h in static['habitat'].dropna().unique()]
    elevation_order = ndvi_get_unique_ordered_elevation_bins_from_static(
        static)
    print('\nHabitat counts assigned to NDVI pixel centers:')
    print(static['habitat'].value_counts(dropna=False))
    print('\nElevation bins assigned to NDVI pixel centers:')
    print(static['elev_bin'].value_counts(dropna=False).sort_index())
    return {'static_pixel_table': static.drop(columns='geometry'), 'habitat_order': habitat_order, 'elevation_order': elevation_order}


def ndvi_make_aggregate_pixel_dataframe(center_map, p_map, land_mask, sd_meta, static_layers, transition_label, metric, group_label):
    delta = center_map.values.astype(float)
    pvals = p_map.values.astype(float)
    land = land_mask.values.astype(bool)
    significant = (pvals < ndvi_ALPHA) & land & np.isfinite(delta)
    sd = sd_meta['sd']
    response_codes = ndvi_response_class_array(delta, significant, sd)
    rows, cols = np.where(land & np.isfinite(delta))
    records = []
    for r, c in zip(rows, cols):
        resp_code = int(response_codes[r, c])
        records.append({'metric': metric, 'variable': ndvi_VARIABLE_NAME, 'group': group_label, 'transition': transition_label, 'row': int(r), 'col': int(c), 'delta_ndvi': float(delta[r, c]), 'p_value': float(
            pvals[r, c]) if np.isfinite(pvals[r, c]) else np.nan, 'significant': bool(significant[r, c]), 'response_code': resp_code, 'response_class': ndvi_RESPONSE_LABELS.get(resp_code, 'Unknown'), 'sd_threshold': float(sd)})
    pixel_df = pd.DataFrame.from_records(records)
    static_df = static_layers['static_pixel_table']
    pixel_df = pixel_df.merge(static_df, on=['row', 'col'], how='left')
    pixel_df['elev_bin'] = pixel_df['elev_bin'].map(
        ndvi_normalize_elevation_bin_label)
    pixel_df = pixel_df.loc[pixel_df['habitat'].notna(
    ) & pixel_df['elev_bin'].notna() & np.isfinite(pixel_df['area_km2'])].copy()
    pixel_df['response_direction'] = 'Not significant'
    pixel_df.loc[pixel_df['significant'] & (
        pixel_df['delta_ndvi'] < 0), 'response_direction'] = 'Any significant decrease'
    pixel_df.loc[pixel_df['significant'] & (
        pixel_df['delta_ndvi'] > 0), 'response_direction'] = 'Any significant increase'
    return pixel_df


def ndvi_build_event_level_summary(event_deltas, land_mask, static_layers):
    static_df = static_layers['static_pixel_table'].copy()
    land = land_mask.values.astype(bool)
    rows, cols = np.where(land)
    land_df = pd.DataFrame({'row': rows.astype(int), 'col': cols.astype(int)})
    static_df = land_df.merge(static_df, on=['row', 'col'], how='left')
    static_df['elev_bin'] = static_df['elev_bin'].map(
        ndvi_normalize_elevation_bin_label)
    static_df = static_df.loc[static_df['habitat'].notna() & static_df['elev_bin'].notna(
    ) & np.isfinite(static_df['area_km2']) & (static_df['area_km2'] > 0)].copy()
    all_records = []
    for ev in event_deltas:
        delta = ev['delta'].values.astype(float)
        values = []
        for _, row in static_df.iterrows():
            r = int(row['row'])
            c = int(row['col'])
            v = delta[r, c]
            if not np.isfinite(v):
                continue
            values.append({'event_id': ev['event_id'], 'transition': ev['transition'], 'K': ev['K'], 'start_date': ev['start_date'], 'end_date': ev['end_date'], 'habitat': row['habitat'],
                          'elev_bin': ndvi_normalize_elevation_bin_label(row['elev_bin']), 'elev_mid_m': row['elev_mid_m'], 'delta_ndvi': float(v), 'area_km2': float(row['area_km2'])})
        ev_df = pd.DataFrame(values)
        if ev_df.empty:
            continue
        grouped = ev_df.groupby(['event_id', 'transition', 'K', 'start_date', 'end_date', 'habitat', 'elev_bin'], dropna=False).apply(lambda g: pd.Series({'n_pixels': len(g), 'group_area_km2': g['area_km2'].sum(
        ), 'mean_delta_ndvi': ndvi_weighted_mean(g['delta_ndvi'], g['area_km2']), 'median_delta_ndvi': float(np.nanmedian(g['delta_ndvi'])), 'mean_elev_m': ndvi_weighted_mean(g['elev_mid_m'], g['area_km2'])})).reset_index()
        all_records.append(grouped)
    if len(all_records) == 0:
        return pd.DataFrame()
    out = pd.concat(all_records, ignore_index=True)
    out['elev_bin'] = out['elev_bin'].map(ndvi_normalize_elevation_bin_label)
    out = out.loc[out['habitat'].notna() & out['elev_bin'].notna() & np.isfinite(
        out['mean_delta_ndvi']) & np.isfinite(out['mean_elev_m'])].copy()
    return out


def ndvi_summarize_response_direction_by_group(pixel_df, group_col):
    df = pixel_df.copy()
    group_area = df.groupby(['transition', group_col], dropna=False)[
        'area_km2'].sum().reset_index(name='group_area_km2')
    affected = df.loc[df['response_direction'] != 'Not significant'].groupby(
        ['transition', group_col, 'response_direction'], dropna=False)['area_km2'].sum().reset_index(name='affected_area_km2')
    out = affected.merge(group_area, on=['transition', group_col], how='left')
    out['percent_of_group_area'] = 100.0 * \
        out['affected_area_km2'] / out['group_area_km2']
    return out


def ndvi_summarize_response_composition_by_group(pixel_df, group_col):
    df = pixel_df.copy()
    group_area = df.groupby(['transition', group_col], dropna=False)[
        'area_km2'].sum().reset_index(name='group_area_km2')
    class_area = df.groupby(['transition', group_col, 'response_class'], dropna=False)[
        'area_km2'].sum().reset_index(name='class_area_km2')
    out = class_area.merge(
        group_area, on=['transition', group_col], how='left')
    out['percent_of_group_area'] = 100.0 * \
        out['class_area_km2'] / out['group_area_km2']
    return out


def ndvi_summarize_response_direction_by_habitat(pixel_df):
    return ndvi_summarize_response_direction_by_group(pixel_df, 'habitat')


def ndvi_summarize_response_composition_by_habitat(pixel_df):
    return ndvi_summarize_response_composition_by_group(pixel_df, 'habitat')


def ndvi_summarize_response_direction_by_elevation(pixel_df):
    return ndvi_summarize_response_direction_by_group(pixel_df, 'elev_bin')


def ndvi_summarize_response_composition_by_elevation(pixel_df):
    return ndvi_summarize_response_composition_by_group(pixel_df, 'elev_bin')


def ndvi_save_mirrored_bar_plot_by_group(direction_summary, out_png, group_order, group_col, group_label, title_text):
    df = direction_summary.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry->Wet', 'Wet_to_Dry': 'Wet->Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        13, max(5, len(group_order) * 0.55)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        plot_df = pd.DataFrame({group_col: group_order})
        dec = sub.loc[sub['response_direction'] == 'Any significant decrease', [
            group_col, 'percent_of_group_area']].rename(columns={'percent_of_group_area': 'decrease_percent'})
        inc = sub.loc[sub['response_direction'] == 'Any significant increase', [
            group_col, 'percent_of_group_area']].rename(columns={'percent_of_group_area': 'increase_percent'})
        plot_df = plot_df.merge(dec, on=group_col, how='left')
        plot_df = plot_df.merge(inc, on=group_col, how='left')
        plot_df['decrease_percent'] = plot_df['decrease_percent'].fillna(0)
        plot_df['increase_percent'] = plot_df['increase_percent'].fillna(0)
        y = np.arange(len(plot_df))
        ax.barh(y, -plot_df['decrease_percent'], label='Any significant decrease',
                color=ndvi_DIRECTION_COLORS['Any significant decrease'], alpha=0.85)
        ax.barh(y, plot_df['increase_percent'], label='Any significant increase',
                color=ndvi_DIRECTION_COLORS['Any significant increase'], alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel(f'Percent of {group_label.lower()} area')
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df[group_col])
        max_val = max(plot_df['decrease_percent'].max(),
                      plot_df['increase_percent'].max(), 1)
        xlim = max(10, np.ceil(max_val / 10) * 10)
        ax.set_xlim(-xlim, xlim)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{abs(x):.0f}' for x in ticks])
        for i, row in plot_df.iterrows():
            if row['decrease_percent'] > 0:
                ax.text(-row['decrease_percent'] - 0.5, i,
                        f"{row['decrease_percent']:.1f}%", va='center', ha='right', fontsize=8)
            if row['increase_percent'] > 0:
                ax.text(row['increase_percent'] + 0.5, i,
                        f"{row['increase_percent']:.1f}%", va='center', ha='left', fontsize=8)
    axes[0].set_ylabel(group_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle(title_text, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved mirrored {group_label.lower()} bar plot: {out_png}')


def ndvi_save_stacked_response_composition_bars_by_group(composition_df, out_png, group_order, group_col, group_label, title_text):
    df = composition_df.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry->Wet', 'Wet_to_Dry': 'Wet->Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        15, max(5, len(group_order) * 0.55)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        pivot = sub.pivot_table(index=group_col, columns='response_class',
                                values='percent_of_group_area', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(
            index=group_order, columns=ndvi_RESPONSE_CLASS_ORDER, fill_value=0)
        y = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for response_class in ndvi_RESPONSE_CLASS_ORDER:
            vals = pivot[response_class].values
            color = ndvi_RESPONSE_COLORS.get(response_class, '#cccccc')
            ax.barh(y, vals, left=left, color=color,
                    label=response_class, edgecolor='white', linewidth=0.4)
            left += vals
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel(f'Percent of {group_label.lower()} area')
        ax.set_xlim(0, 100)
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
    axes[0].set_ylabel(group_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(title_text, fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.15, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved stacked {group_label.lower()} composition bars: {out_png}')


def ndvi_save_mirrored_bar_plot_by_habitat(direction_summary, out_png, habitat_order):
    ndvi_save_mirrored_bar_plot_by_group(direction_summary=direction_summary, out_png=out_png, group_order=habitat_order, group_col='habitat', group_label='Habitat',
                                         title_text='Significant Delta NDVI response footprint by habitat\nMirrored bars show percent of habitat area with significant decrease or increase')


def ndvi_save_mirrored_bar_plot_by_elevation(direction_summary, out_png, elevation_order):
    ndvi_save_mirrored_bar_plot_by_group(direction_summary=direction_summary, out_png=out_png, group_order=elevation_order, group_col='elev_bin', group_label='Elevation bin',
                                         title_text='Significant Delta NDVI response footprint by elevation bin\nMirrored bars show percent of elevation-bin area with significant decrease or increase')


def ndvi_save_stacked_response_composition_bars(composition_df, out_png, habitat_order):
    ndvi_save_stacked_response_composition_bars_by_group(composition_df=composition_df, out_png=out_png, group_order=habitat_order, group_col='habitat',
                                                         group_label='Habitat', title_text='Habitat response composition during whiplash events\nBars match the Delta NDVI map classes')


def ndvi_save_stacked_response_composition_bars_by_elevation(composition_df, out_png, elevation_order):
    ndvi_save_stacked_response_composition_bars_by_group(composition_df=composition_df, out_png=out_png, group_order=elevation_order, group_col='elev_bin',
                                                         group_label='Elevation bin', title_text='Elevation-bin response composition during whiplash events\nBars match the Delta NDVI map classes')


def ndvi_save_stacked_response_composition_bars_with_model_stars(composition_df, out_png, habitat_order):
    df = composition_df.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry->Wet', 'Wet_to_Dry': 'Wet->Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        16, max(5.5, len(habitat_order) * 0.58)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        pivot = sub.pivot_table(index='habitat', columns='response_class',
                                values='percent_of_group_area', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(index=habitat_order,
                              columns=ndvi_RESPONSE_CLASS_ORDER, fill_value=0)
        y = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for response_class in ndvi_RESPONSE_CLASS_ORDER:
            vals = pivot[response_class].values
            color = ndvi_RESPONSE_COLORS.get(response_class, '#cccccc')
            ax.barh(y, vals, left=left, color=color,
                    label=response_class, edgecolor='white', linewidth=0.4)
            left += vals
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel('Percent of habitat area')
        ax.set_xlim(0, 108)
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
        stars_for_transition = ndvi_SIGNIFICANCE_STARS.get(transition, {})
        for i, habitat in enumerate(pivot.index):
            stars = stars_for_transition.get(habitat, '')
            if stars:
                ax.text(102.0, i, stars, va='center', ha='left',
                        fontsize=12, fontweight='bold', color='black')
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel('Habitat')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle('Habitat response composition during whiplash events\nBars match the Delta NDVI map classes, stars show event-level model significance', fontsize=15, y=0.98)
    fig.text(0.5, 0.025, ndvi_SIGNIFICANCE_NOTE,
             ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, 0.17, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(
        f'Saved stacked habitat composition bars with model stars: {out_png}')


def ndvi_save_model_results(model, out_txt, title):
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write('=' * len(title) + '\n\n')
        f.write(str(model.summary()))


def ndvi_model_to_coef_table(model):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    bse = model.bse
    out = pd.DataFrame({'term': params.index, 'estimate': params.values, 'std_error': bse.values,
                       'p_value': pvals.values, 'ci_lower': conf[0].values, 'ci_upper': conf[1].values})
    return out


def ndvi_fit_wls_cluster(formula, df, weight_col='group_area_km2', cluster_col='event_id'):
    model = smf.wls(formula=formula, data=df, weights=df[weight_col]).fit(
        cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
    return model


def ndvi_run_event_level_statistics(event_summary_df, out_dir):
    stats_dir = Path(out_dir) / 'Event_Level_Statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    df = event_summary_df.copy()
    df['elev_bin'] = df['elev_bin'].map(ndvi_normalize_elevation_bin_label)
    df = df.loc[df['habitat'].notna() & df['elev_bin'].notna() & df['transition'].notna() & np.isfinite(
        df['mean_delta_ndvi']) & np.isfinite(df['mean_elev_m']) & np.isfinite(df['group_area_km2']) & (df['group_area_km2'] > 0)].copy()
    df['transition'] = pd.Categorical(df['transition'], categories=[
                                      'Dry_to_Wet', 'Wet_to_Dry'], ordered=False)
    habitat_categories = ['Shrub', 'Agriculture', 'Conifer_Forest',
                          'Conifer_Woodland', 'Desert', 'Hardwood', 'Herbaceous', 'Urban']
    habitat_categories = [
        h for h in habitat_categories if h in df['habitat'].dropna().unique()]
    if ndvi_REFERENCE_HABITAT not in habitat_categories:
        raise ValueError(
            f"Reference habitat '{ndvi_REFERENCE_HABITAT}' not found in model data. Available habitats are: {habitat_categories}")
    df['habitat'] = pd.Categorical(
        df['habitat'], categories=habitat_categories, ordered=False)
    elevation_categories = ndvi_get_unique_ordered_elevation_bins_from_event_df(
        df)
    if ndvi_REFERENCE_ELEV_BIN not in elevation_categories:
        raise ValueError(
            f"Reference elevation bin '{ndvi_REFERENCE_ELEV_BIN}' not found in model data. Available elevation bins are: {elevation_categories}")
    df['elev_bin'] = pd.Categorical(
        df['elev_bin'], categories=elevation_categories, ordered=True)
    df.to_csv(stats_dir / 'event_level_model_input.csv', index=False)
    model_specs = {'M1_transition_only': "mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet'))", 'M2_habitat_only': "mean_delta_ndvi ~ C(habitat, Treatment(reference='Shrub'))", 'M3_elevation_continuous_only': 'mean_delta_ndvi ~ mean_elev_m', 'M4_transition_habitat': "mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(habitat, Treatment(reference='Shrub'))", 'M5_transition_elevation_continuous': "mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet')) * mean_elev_m", 'M6_full_transition_habitat_elevation_continuous': "mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(habitat, Treatment(reference='Shrub')) + C(transition, Treatment(reference='Dry_to_Wet')) * mean_elev_m",
                   'M7_elevation_bin_only': f"mean_delta_ndvi ~ C(elev_bin, Treatment(reference='{ndvi_REFERENCE_ELEV_BIN}'))", 'M8_transition_elevation_bin': f"mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(elev_bin, Treatment(reference='{ndvi_REFERENCE_ELEV_BIN}'))", 'M9_full_habitat_elevation_bin': f"mean_delta_ndvi ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(habitat, Treatment(reference='Shrub')) + C(transition, Treatment(reference='Dry_to_Wet')) * C(elev_bin, Treatment(reference='{ndvi_REFERENCE_ELEV_BIN}'))"}
    model_summaries = []
    for name, formula in model_specs.items():
        try:
            model = ndvi_fit_wls_cluster(formula, df)
            ndvi_save_model_results(
                model, stats_dir / f"{name}_SHRUB_REFERENCE_ELEV_{ndvi_REFERENCE_ELEV_BIN.replace(' ', '_').replace('-', '_')}_cluster_robust_summary.txt", title=f'{name}: {formula}')
            coef_table = ndvi_model_to_coef_table(model)
            coef_table.insert(0, 'variable', ndvi_VARIABLE_NAME)
            coef_table.insert(1, 'reference_transition',
                              ndvi_REFERENCE_TRANSITION)
            coef_table.insert(2, 'reference_habitat', ndvi_REFERENCE_HABITAT)
            coef_table.insert(3, 'reference_elevation_bin',
                              ndvi_REFERENCE_ELEV_BIN)
            coef_table.to_csv(
                stats_dir / f"{name}_SHRUB_REFERENCE_ELEV_{ndvi_REFERENCE_ELEV_BIN.replace(' ', '_').replace('-', '_')}_cluster_robust_coefficients.csv", index=False)
            model_summaries.append({'variable': ndvi_VARIABLE_NAME, 'model': name, 'formula': formula, 'reference_transition': ndvi_REFERENCE_TRANSITION, 'reference_habitat': ndvi_REFERENCE_HABITAT,
                                   'reference_elevation_bin': ndvi_REFERENCE_ELEV_BIN, 'n_rows': int(model.nobs), 'r_squared': getattr(model, 'rsquared', np.nan), 'aic': getattr(model, 'aic', np.nan), 'bic': getattr(model, 'bic', np.nan)})
        except Exception as e:
            model_summaries.append({'variable': ndvi_VARIABLE_NAME, 'model': name, 'formula': formula, 'reference_transition': ndvi_REFERENCE_TRANSITION,
                                   'reference_habitat': ndvi_REFERENCE_HABITAT, 'reference_elevation_bin': ndvi_REFERENCE_ELEV_BIN, 'error': str(e)})
    pd.DataFrame(model_summaries).to_csv(
        stats_dir / 'model_summary_table_SHRUB_REFERENCE_ELEVATION_BIN_REFERENCE.csv', index=False)
    kruskal_records = []
    try:
        hab_groups = [g['mean_delta_ndvi'].dropna().values for _, g in df.groupby(
            'habitat', observed=True) if len(g['mean_delta_ndvi'].dropna()) >= 2]
        if len(hab_groups) >= 2:
            H, p = kruskal(*hab_groups)
            kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis habitat difference', 'statistic': H,
                                   'p_value': p, 'reference_habitat': ndvi_REFERENCE_HABITAT, 'note': 'Uses event-level habitat-elevation summaries.'})
    except Exception as e:
        kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis habitat difference',
                               'reference_habitat': ndvi_REFERENCE_HABITAT, 'error': str(e)})
    try:
        elev_groups = [g['mean_delta_ndvi'].dropna().values for _, g in df.groupby(
            'elev_bin', observed=True) if len(g['mean_delta_ndvi'].dropna()) >= 2]
        if len(elev_groups) >= 2:
            H, p = kruskal(*elev_groups)
            kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis elevation-bin difference', 'statistic': H,
                                   'p_value': p, 'reference_elevation_bin': ndvi_REFERENCE_ELEV_BIN, 'note': 'Uses event-level elevation-bin summaries.'})
    except Exception as e:
        kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis elevation-bin difference',
                               'reference_elevation_bin': ndvi_REFERENCE_ELEV_BIN, 'error': str(e)})
    try:
        trans_groups = [g['mean_delta_ndvi'].dropna().values for _, g in df.groupby(
            'transition', observed=True) if len(g['mean_delta_ndvi'].dropna()) >= 2]
        if len(trans_groups) == 2:
            H, p = kruskal(*trans_groups)
            kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis transition difference', 'statistic': H, 'p_value': p,
                                   'reference_transition': ndvi_REFERENCE_TRANSITION, 'note': 'Tests Dry_to_Wet vs Wet_to_Dry using event-level summaries.'})
    except Exception as e:
        kruskal_records.append({'variable': ndvi_VARIABLE_NAME, 'test': 'Kruskal-Wallis transition difference',
                               'reference_transition': ndvi_REFERENCE_TRANSITION, 'error': str(e)})
    pd.DataFrame(kruskal_records).to_csv(
        stats_dir / 'nonparametric_tests_SHRUB_AND_ELEVATION_BIN_REFERENCE.csv', index=False)
    print(f'Saved event-level statistics to: {stats_dir}')
    print('Reference habitat for model coefficients: Shrub')
    print('Reference transition for model coefficients: Dry_to_Wet')
    print(
        f'Reference elevation bin for model coefficients: {ndvi_REFERENCE_ELEV_BIN}')


def ndvi_save_rasters(center_map, p_map, out_dir, out_name_base):
    if not ndvi_SAVE_RASTERS:
        return
    raster_dir = Path(out_dir) / 'GeoTIFFs'
    raster_dir.mkdir(parents=True, exist_ok=True)
    center_map = ndvi_ensure_spatial_dims(center_map)
    p_map = ndvi_ensure_spatial_dims(p_map)
    center_map.rio.to_raster(
        raster_dir / f'{out_name_base}_center_delta_ndvi.tif')
    p_map.rio.to_raster(raster_dir / f'{out_name_base}_wilcoxon_p.tif')
    print(f'Saved GeoTIFFs to: {raster_dir}')


def ndvi_process_direction(ndvi, soc, land_mask, df_events, direction, out_dir, out_name_base, title_text_base, static_layers, group_label):
    if len(df_events) < ndvi_MIN_EVENTS:
        print(f'Skipping {direction}, not enough events: {len(df_events)}')
        return (None, None)
    event_deltas = ndvi_compute_event_deltas(ndvi, df_events)
    if len(event_deltas) < ndvi_MIN_EVENTS:
        print(
            f'Skipping {direction}, not enough valid deltas: {len(event_deltas)}')
        return (None, None)
    stack = xr.concat([ev['delta'] for ev in event_deltas],
                      dim='event', coords='minimal', compat='override')
    center_map, p_map = xr.apply_ufunc(lambda x: ndvi_run_wilcoxon(x), stack, input_core_dims=[
                                       ['event']], output_core_dims=[[], []], vectorize=True, output_dtypes=[float, float])
    center_map = center_map.rio.write_crs(ndvi.rio.crs)
    p_map = p_map.rio.write_crs(ndvi.rio.crs)
    center_map = ndvi_ensure_spatial_dims(center_map)
    p_map = ndvi_ensure_spatial_dims(p_map)
    sig_mask = (p_map < ndvi_ALPHA) & land_mask
    out_dir.mkdir(parents=True, exist_ok=True)
    ndvi_save_rasters(center_map, p_map, out_dir, out_name_base)
    sd_map, sd_meta = ndvi_classify_sd_map(center_map, sig_mask, land_mask)
    if sd_meta is None:
        print(f'No valid SD map for {direction}')
        return (None, None)
    ndvi_save_single_map(class_map=sd_map, soc=soc, out_file=out_dir / f'{out_name_base}_PlusMinus1SD.png',
                         title_text=f'GIMMS Delta NDVI ({ndvi_METRIC}) | {direction}\n{title_text_base}\nSignificant Pixels Only (Wilcoxon p < {ndvi_ALPHA}) | Data-driven bins, +/-1 SD', meta=sd_meta)
    transition_label = direction.replace('->', '_to_')
    aggregate_pixel_df = ndvi_make_aggregate_pixel_dataframe(center_map=center_map, p_map=p_map, land_mask=land_mask, sd_meta=sd_meta,
                                                             static_layers=static_layers, transition_label=transition_label, metric=ndvi_METRIC, group_label=group_label)
    event_summary_df = ndvi_build_event_level_summary(
        event_deltas=event_deltas, land_mask=land_mask, static_layers=static_layers)
    transition_dir = Path(out_dir) / \
        f'Habitat_Elevation_Stats_{transition_label}'
    transition_dir.mkdir(parents=True, exist_ok=True)
    aggregate_pixel_df.to_csv(
        transition_dir / 'aggregate_pixel_table_for_figures.csv', index=False)
    event_summary_df.to_csv(
        transition_dir / 'event_level_habitat_elevation_summary.csv', index=False)
    return (aggregate_pixel_df, event_summary_df)


def ndvi_prepare_ndvi_common_data():
    print('\nLoading common NDVI data for all three detection methods...')
    soc = gpd.read_file(ndvi_SOCAL_SHP)
    ds = ndvi_open_gimms_dataset(ndvi_GIMMS_DIR)
    ndvi_candidates = [v for v in ds.data_vars if 'NDVI' in v.upper()]
    if len(ndvi_candidates) == 0:
        raise ValueError('No NDVI variable found in the GIMMS dataset.')
    ndvi_var = ndvi_candidates[0]
    print(f'Using NDVI variable: {ndvi_var}')
    ndvi = ds[ndvi_var]
    ndvi = ndvi.where(ndvi > -2000) * 0.0001
    ndvi = ndvi.rio.write_crs('EPSG:4326', inplace=False)
    ndvi = ndvi_ensure_spatial_dims(ndvi)
    if 'time' not in ndvi.dims:
        raise ValueError('NDVI data must have a time dimension.')
    time_idx = ndvi.indexes['time']
    if hasattr(time_idx, 'to_datetimeindex'):
        time_index = time_idx.to_datetimeindex()
    else:
        time_index = pd.DatetimeIndex(pd.to_datetime(time_idx))
    keep = (time_index >= ndvi_ANALYSIS_START) & (
        time_index <= ndvi_ANALYSIS_END)
    if not np.any(keep):
        raise ValueError(
            'NDVI contains no observations from January 1982 through December 2022.')
    ndvi = ndvi.isel(time=np.flatnonzero(keep))
    time_index = pd.DatetimeIndex(time_index[keep])
    ndvi = ndvi.rio.clip(soc.geometry, soc.crs, drop=True).load()
    ndvi = ndvi_ensure_spatial_dims(ndvi)
    ndvi = ndvi.assign_coords(ym=('time', time_index.strftime('%Y-%m')))
    print(
        f'NDVI analysis period: {time_index.min():%Y-%m} to {time_index.max():%Y-%m}; {len(time_index):,} time steps.')
    land_mask = xr.DataArray(np.isfinite(ndvi.isel(time=0).values), coords=ndvi.isel(
        time=0).coords, dims=ndvi.isel(time=0).dims)
    land_mask = land_mask.rio.write_crs(ndvi.rio.crs)
    land_mask = ndvi_ensure_spatial_dims(land_mask)
    static_layers = ndvi_prepare_static_layers(ndvi.isel(time=0))
    return {'soc': soc, 'ndvi': ndvi, 'land_mask': land_mask, 'static_layers': static_layers}


def ndvi_run_ndvi_detection_method(common_data, detection_method):
    print('\n' + '#' * 79)
    print(f'NDVI WHIPLASH DETECTION METHOD: {detection_method}')
    print('#' * 79)
    soc = common_data['soc']
    ndvi = common_data['ndvi']
    land_mask = common_data['land_mask']
    static_layers = common_data['static_layers']
    df_all_k = ndvi_load_whiplash_events(detection_method)
    out_dir = ndvi_BASE_OUT_ROOT / detection_method / \
        ndvi_VARIABLE_NAME / ndvi_METRIC / 'All_K' / 'RW_3-Month_Mean'
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_tables = []
    event_summary_tables = []
    for direction in ndvi_DIRECTIONS:
        df_combo = df_all_k[df_all_k['Direction'] == direction].copy()
        if len(df_combo) < ndvi_MIN_EVENTS:
            print(f'Skipping {direction}, not enough events: {len(df_combo)}')
            continue
        out_name_base = f"GIMMS_{ndvi_METRIC}_AllK_{direction.replace('->', 'To')}"
        title_text = f'All K Combined, Window=3-Month Mean, Events={len(df_combo)}'
        print(f'\nProcessing {direction}, events={len(df_combo)}')
        agg_df, event_df = ndvi_process_direction(ndvi=ndvi, soc=soc, land_mask=land_mask, df_events=df_combo, direction=direction,
                                                  out_dir=out_dir, out_name_base=out_name_base, title_text_base=title_text, static_layers=static_layers, group_label='All_K')
        if agg_df is not None:
            aggregate_tables.append(agg_df)
        if event_df is not None:
            event_summary_tables.append(event_df)
    if len(aggregate_tables) > 0:
        aggregate_all = pd.concat(aggregate_tables, ignore_index=True)
        aggregate_all['elev_bin'] = aggregate_all['elev_bin'].map(
            ndvi_normalize_elevation_bin_label)
        figures_dir = out_dir / 'Publishable_Figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        aggregate_all.to_csv(
            figures_dir / 'combined_aggregate_pixel_table_for_figures.csv', index=False)
        habitat_direction_summary = ndvi_summarize_response_direction_by_habitat(
            aggregate_all)
        habitat_composition_summary = ndvi_summarize_response_composition_by_habitat(
            aggregate_all)
        elevation_direction_summary = ndvi_summarize_response_direction_by_elevation(
            aggregate_all)
        elevation_composition_summary = ndvi_summarize_response_composition_by_elevation(
            aggregate_all)
        habitat_direction_summary.to_csv(
            figures_dir / 'summary_any_significant_decrease_increase_by_habitat.csv', index=False)
        habitat_composition_summary.to_csv(
            figures_dir / 'summary_response_composition_by_habitat.csv', index=False)
        elevation_direction_summary.to_csv(
            figures_dir / 'summary_any_significant_decrease_increase_by_elevation.csv', index=False)
        elevation_composition_summary.to_csv(
            figures_dir / 'summary_response_composition_by_elevation.csv', index=False)
        habitat_order_for_figs = [
            h for h in ndvi_HABITAT_ORDER if h in aggregate_all['habitat'].dropna().unique()]
        elevation_order_for_figs = [
            e for e in static_layers['elevation_order'] if e in aggregate_all['elev_bin'].dropna().unique()]
        ndvi_save_mirrored_bar_plot_by_habitat(direction_summary=habitat_direction_summary, out_png=figures_dir /
                                               'figure_mirrored_bar_any_significant_response_by_habitat.png', habitat_order=habitat_order_for_figs)
        ndvi_save_stacked_response_composition_bars(composition_df=habitat_composition_summary, out_png=figures_dir /
                                                    'figure_stacked_response_composition_by_habitat.png', habitat_order=habitat_order_for_figs)
        ndvi_save_stacked_response_composition_bars_with_model_stars(
            composition_df=habitat_composition_summary, out_png=figures_dir / 'figure_stacked_response_composition_by_habitat_WITH_MODEL_STARS.png', habitat_order=habitat_order_for_figs)
        ndvi_save_mirrored_bar_plot_by_elevation(direction_summary=elevation_direction_summary, out_png=figures_dir /
                                                 'figure_mirrored_bar_any_significant_response_by_elevation.png', elevation_order=elevation_order_for_figs)
        ndvi_save_stacked_response_composition_bars_by_elevation(
            composition_df=elevation_composition_summary, out_png=figures_dir / 'figure_stacked_response_composition_by_elevation.png', elevation_order=elevation_order_for_figs)
    if len(event_summary_tables) > 0:
        event_all = pd.concat(event_summary_tables, ignore_index=True)
        event_all['elev_bin'] = event_all['elev_bin'].map(
            ndvi_normalize_elevation_bin_label)
        event_all.to_csv(
            out_dir / 'combined_event_level_habitat_elevation_summary.csv', index=False)
        if ndvi_SAVE_STATS:
            ndvi_run_event_level_statistics(event_all, out_dir)
    print('\nFinished.')
    print(f'Main output folder:\n{out_dir}')
    print('\nKey outputs:')
    print('  Publishable_Figures/figure_mirrored_bar_any_significant_response_by_habitat.png')
    print('  Publishable_Figures/figure_stacked_response_composition_by_habitat.png')
    print('  Publishable_Figures/figure_stacked_response_composition_by_habitat_WITH_MODEL_STARS.png')
    print('  Publishable_Figures/figure_mirrored_bar_any_significant_response_by_elevation.png')
    print('  Publishable_Figures/figure_stacked_response_composition_by_elevation.png')
    print('  Event_Level_Statistics/model_summary_table_SHRUB_REFERENCE_ELEVATION_BIN_REFERENCE.csv')
    print('  Event_Level_Statistics/M8_transition_elevation_bin_SHRUB_REFERENCE_ELEV_0_250_m_cluster_robust_coefficients.csv')
    print('  Event_Level_Statistics/M9_full_habitat_elevation_bin_SHRUB_REFERENCE_ELEV_0_250_m_cluster_robust_coefficients.csv')
    print('  Event_Level_Statistics/nonparametric_tests_SHRUB_AND_ELEVATION_BIN_REFERENCE.csv')


def ndvi_main():
    print('\nStarting combined NDVI response analysis for all three whiplash detection methods.')
    print(
        f'Required analysis period: {ndvi_ANALYSIS_START:%Y-%m} through {ndvi_ANALYSIS_END:%Y-%m}')
    common_data = ndvi_prepare_ndvi_common_data()
    for detection_method in ndvi_DETECTION_METHODS:
        ndvi_run_ndvi_detection_method(common_data, detection_method)
    print('\nFinished all NDVI detection-method analyses.')


smrz_FINAL_PAPER_ROOT = Path('D:\\Graduate Research\\Final_Paper_Analysis')
smrz_BASE_OUT_ROOT = smrz_FINAL_PAPER_ROOT / \
    'Whiplash_Response_NDVI_SMRz_1982_2022'
smrz_WHIPLASH_ROOT = smrz_FINAL_PAPER_ROOT / \
    'Whiplash Analysis Precipitation 1982_2022'
smrz_SMRZ_DIR = 'D:\\Graduate Research\\Graduate Research\\Data\\GLEAM\\SMrz_GlEAM_Data'
smrz_SOCAL_SHP = 'D:\\Graduate Research\\Graduate Research\\Data\\BoundingBox\\BoundingBox.shp'
smrz_VEG_SHPS = {'Agriculture': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Agriculturen_4km.shp', 'Conifer_Forest': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Conifer_Forest_4km.shp', 'Conifer_Woodland': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Conifer_Woodland_4km.shp', 'Desert': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Desert_4km.shp',
                 'Hardwood': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Hardwood_4km.shp', 'Herbaceous': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Herbaceous_4km.shp', 'Shrub': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Shrub_4km.shp', 'Urban': 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Updated_Vegetation\\4km_Vegetation_Types\\Urban_4km.shp'}
smrz_ELEV_SHP = 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Habitats_Elevation\\Elevation_Different_Resolutions\\elevation_classified_8km_250m_bins_land_northup.shp'
smrz_ELEV_SHP_FALLBACK = 'D:\\Graduate Research\\Graduate Research\\California Vegetation - WHR13 Types ArcGIS Pro\\Habitats_Elevation\\Elevation_Different_Resolutions\\elevation_classified_1km_250m_bins_land_northup.shp'
smrz_VARIABLE_NAME = 'SMrz'
smrz_METRIC = 'MEAN'
smrz_K_LIST = [3, 6, 12, 24]
smrz_TARGET_WINDOW = '3-Month Mean'
smrz_DIRECTIONS = ['Dry->Wet', 'Wet->Dry']
smrz_BUFFER_MONTHS = 3
smrz_MIN_EVENTS = 6
smrz_ALPHA = 0.05
smrz_ELEV_BIN_FIELD = None
smrz_ELEV_MIDPOINT_FIELD = None
smrz_ELEV_BIN_SIZE_M = 250
smrz_SAVE_RASTERS = True
smrz_SAVE_STATS = True
smrz_HABITAT_ORDER = list(smrz_VEG_SHPS.keys())
smrz_REFERENCE_HABITAT = 'Shrub'
smrz_REFERENCE_TRANSITION = 'Dry_to_Wet'
smrz_ANALYSIS_START = pd.Timestamp('1982-01-01')
smrz_ANALYSIS_END = pd.Timestamp('2022-12-31')
smrz_DETECTION_METHODS = ['Original_MultiK',
                          'Chronological_First_Opposite', 'Shortest_First']
smrz_K_FILE_LABELS = {3: '1-3 months', 6: '4-6 months',
                      12: '7-12 months', 24: '13-24 months'}


def smrz_get_whiplash_event_path(detection_method, k_value):
    """Return the exact K-specific SD_0.8 CSV created by the detection script."""
    if detection_method not in smrz_DETECTION_METHODS:
        raise ValueError(
            f'Unknown detection method: {detection_method}. Expected one of: {smrz_DETECTION_METHODS}')
    if k_value not in smrz_K_FILE_LABELS:
        raise ValueError(f'Unsupported K value: {k_value}')
    filename = f'Whiplash_Events_{detection_method}_SD_0.8_K{k_value}_{smrz_K_FILE_LABELS[k_value]}.csv'
    return smrz_WHIPLASH_ROOT / detection_method / 'SD_0.8' / filename


def smrz_load_whiplash_events(detection_method):
    """
    Load all four exact K-specific event files for one detection method.

    Files are required to exist, including an empty K24 file where applicable.
    Events are restricted to the 3-Month Mean precipitation series and to
    January 1982 through December 2022.
    """
    expected_paths = {k_value: smrz_get_whiplash_event_path(
        detection_method, k_value) for k_value in smrz_K_LIST}
    missing_paths = [path for path in expected_paths.values()
                     if not path.exists()]
    if missing_paths:
        missing_text = '\n'.join((f'  - {path}' for path in missing_paths))
        raise FileNotFoundError(
            f'Missing required whiplash event file(s) for {detection_method}:\n{missing_text}')
    all_k_frames = []
    for k_value in smrz_K_LIST:
        df_path = expected_paths[k_value]
        print(f'Using K{k_value} event file: {df_path}')
        df_all = pd.read_csv(df_path)
        required_columns = {'StartDate', 'EndDate', 'Direction'}
        missing_columns = sorted(required_columns - set(df_all.columns))
        if missing_columns:
            raise ValueError(
                f'Event file is missing required columns {missing_columns}: {df_path}')
        df_all['StartDate'] = pd.to_datetime(
            df_all['StartDate'], errors='coerce')
        df_all['EndDate'] = pd.to_datetime(df_all['EndDate'], errors='coerce')
        df_all = df_all.dropna(
            subset=['StartDate', 'EndDate', 'Direction']).copy()
        if 'Window' in df_all.columns:
            df_all = df_all.loc[df_all['Window'].eq(smrz_TARGET_WINDOW)].copy()
        df_all = df_all.loc[df_all['StartDate'].ge(
            smrz_ANALYSIS_START) & df_all['EndDate'].le(smrz_ANALYSIS_END)].copy()
        df_all['K'] = int(k_value)
        if not df_all.empty:
            all_k_frames.append(df_all)
        print(
            f'  Loaded {len(df_all):,} usable K{k_value} events for {detection_method}.')
    if not all_k_frames:
        raise ValueError(
            f'No usable January 1982-December 2022 events remained for {detection_method} and Window={smrz_TARGET_WINDOW}.')
    df_all_k = pd.concat(all_k_frames, ignore_index=True)
    df_all_k = df_all_k.sort_values(
        ['StartDate', 'EndDate', 'K', 'Direction']).reset_index(drop=True)
    print(f'\nCombined All K event counts for {detection_method}:')
    print(df_all_k['Direction'].value_counts().to_string())
    print('\nCounts by K:')
    print(df_all_k['K'].value_counts().sort_index().to_string())
    return df_all_k


smrz_RESPONSE_COLORS = {'Strong decrease (< -1 SD)': '#d7191c', 'Moderate decrease (-1 SD to 0)': '#fdae61', 'Not significant': '#d9d9d9',
                        'Zero significant change': '#bdbdbd', 'Moderate increase (0 to +1 SD)': '#a6d96a', 'Strong increase (> +1 SD)': '#1a9641'}
smrz_RESPONSE_CLASS_ORDER = ['Strong decrease (< -1 SD)', 'Moderate decrease (-1 SD to 0)',
                             'Not significant', 'Moderate increase (0 to +1 SD)', 'Strong increase (> +1 SD)']
smrz_DIRECTION_COLORS = {
    'Any significant decrease': '#d7191c', 'Any significant increase': '#1a9641'}
smrz_cmap_5 = ListedColormap(
    ['#d7191c', '#fdae61', '#d9d9d9', '#a6d96a', '#1a9641'])
smrz_RESPONSE_LABELS = {0: 'Strong decrease (< -1 SD)', 1: 'Moderate decrease (-1 SD to 0)', 2: 'Zero significant change',
                        3: 'Moderate increase (0 to +1 SD)', 4: 'Strong increase (> +1 SD)', 9: 'Not significant'}
smrz_CITIES = {'Los Angeles': (-118.2437, 34.0522), 'San Diego': (-117.1611, 32.7157), 'Santa Barbara': (-119.6982,
                                                                                                         34.4208), 'Palm Springs': (-116.5453, 33.8303), 'Lancaster': (-118.1542, 34.6868), 'Oxnard': (-119.1771, 34.1975)}


def smrz_expand_months(date):
    s = date - relativedelta(months=smrz_BUFFER_MONTHS)
    e = date + relativedelta(months=smrz_BUFFER_MONTHS)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime('%Y-%m'))
        cur += relativedelta(months=1)
    return out


def smrz_run_wilcoxon(x):
    x = x[~np.isnan(x)]
    if len(x) < smrz_MIN_EVENTS or np.all(x == 0):
        return (np.nan, 1.0)
    try:
        _, p = wilcoxon(x)
        return (np.nanmean(x), p)
    except Exception:
        return (np.nanmean(x), 1.0)


def smrz_clean_geometries(gdf):
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    try:
        gdf['geometry'] = gdf.geometry.make_valid()
    except Exception:
        gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def smrz_ensure_spatial_dims(da):
    possible_x = ['x', 'lon', 'longitude']
    possible_y = ['y', 'lat', 'latitude']
    x_dim = None
    y_dim = None
    for d in da.dims:
        if d.lower() in possible_x:
            x_dim = d
        if d.lower() in possible_y:
            y_dim = d
    if x_dim is not None and y_dim is not None:
        da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326', inplace=False)
    return da


def smrz_weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def smrz_parse_midpoint_from_bin_label(label):
    if pd.isna(label):
        return np.nan
    s = str(label).strip().replace(',', '')
    s = re.sub('\\s+', ' ', s)
    if re.fullmatch('\\d+', s):
        code = int(s)
        low = code * smrz_ELEV_BIN_SIZE_M
        high = (code + 1) * smrz_ELEV_BIN_SIZE_M
        return (low + high) / 2.0
    nums = re.findall('-?\\d+\\.?\\d*', s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def smrz_normalize_elevation_bin_label(label):
    """Return clean positive elevation-bin labels such as 0-250 m.

    Handles class IDs like 0, 1, 2; labels like 0 m, 1 m, 2 m;
    and range labels such as 0-250 m. Minus signs are treated as range
    separators so accidental labels like -250-0 m become 0-250 m.
    """
    if pd.isna(label):
        return 'Unknown'
    s = str(label).strip().replace(',', '')
    s = s.replace('–', '-').replace('—', '-')
    s = re.sub(r'\s+', ' ', s)

    class_match = re.fullmatch(r'(-?\d+(?:\.0+)?)\s*m?', s)
    if class_match:
        val = float(class_match.group(1))
        if val.is_integer() and 0 <= int(val) <= 50:
            code = int(val)
            low = code * smrz_ELEV_BIN_SIZE_M
            high = (code + 1) * smrz_ELEV_BIN_SIZE_M
            return f'{low}-{high} m'

    nums = re.findall(r'\d+\.?\d*', s)
    if len(nums) >= 2:
        low = float(nums[0])
        high = float(nums[1])
        low, high = sorted([low, high])
        if low.is_integer():
            low = int(low)
        if high.is_integer():
            high = int(high)
        return f'{low}-{high} m'
    return s.replace('--', '-')


def smrz_make_elevation_label(label):
    return smrz_normalize_elevation_bin_label(label)


def smrz_choose_elevation_bin_field(elev_gdf):
    if smrz_ELEV_BIN_FIELD is not None:
        if smrz_ELEV_BIN_FIELD not in elev_gdf.columns:
            raise ValueError(
                f"ELEV_BIN_FIELD '{smrz_ELEV_BIN_FIELD}' was not found.\nAvailable fields: {list(elev_gdf.columns)}")
        return smrz_ELEV_BIN_FIELD
    candidates = ['elev_bin', 'ELEV_BIN', 'Elevation_Bin', 'elevation_bin', 'class', 'Class',
                  'CLASS', 'gridcode', 'GRIDCODE', 'grid_code', 'GRID_CODE', 'bin', 'BIN', 'Value', 'VALUE', 'DN']
    for c in candidates:
        if c in elev_gdf.columns:
            return c
    for c in elev_gdf.columns:
        cl = c.lower()
        if 'elev' in cl or 'bin' in cl or 'class' in cl or ('grid' in cl):
            return c
    raise ValueError(
        f'Could not detect the elevation-bin field automatically.\nAvailable elevation fields are:\n{list(elev_gdf.columns)}\nSet ELEV_BIN_FIELD near the top of the script.')


def smrz_find_smrz_variable(ds):
    data_vars = list(ds.data_vars)
    preferred = ['SMrz', 'SMRZ', 'smrz', 'SMroot', 'SMrootzone',
                 'root_zone_soil_moisture', 'RootZoneSoilMoisture']
    for p in preferred:
        if p in data_vars:
            return p
    for v in data_vars:
        vu = v.upper()
        if 'SMRZ' in vu:
            return v
        if 'ROOT' in vu and 'SM' in vu:
            return v
        if 'SOIL' in vu and 'MOIST' in vu and ('ROOT' in vu):
            return v
    raise ValueError(
        f'Could not detect SMrz variable automatically.\nAvailable variables are: {data_vars}')


def smrz_open_smrz_dataset(smrz_path):
    if os.path.isfile(smrz_path):
        ds = xr.open_dataset(smrz_path)
    elif os.path.isdir(smrz_path):
        nc_files = sorted(glob.glob(os.path.join(smrz_path, '*.nc*')))
        if len(nc_files) == 0:
            raise FileNotFoundError(
                f'No .nc files found inside SMrz directory: {smrz_path}')
        ds = xr.open_mfdataset(nc_files, combine='by_coords',
                               data_vars='minimal', coords='minimal', compat='override')
    else:
        raise FileNotFoundError(f'SMrz path not found: {smrz_path}')
    var_name = smrz_find_smrz_variable(ds)
    print(f'Using SMrz variable: {var_name}')
    da = ds[var_name]
    allowed_dims = ['time', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude']
    for dim in list(da.dims):
        if dim not in allowed_dims:
            print(f'Selecting first index for extra SMrz dimension: {dim}')
            da = da.isel({dim: 0})
    lon_name = None
    for candidate in ['lon', 'longitude', 'x']:
        if candidate in da.coords or candidate in da.dims:
            lon_name = candidate
            break
    if lon_name is not None:
        lon_vals = da[lon_name].values
        if np.nanmax(lon_vals) > 180:
            da = da.assign_coords({lon_name: (da[lon_name] + 180) % 360 - 180})
            da = da.sortby(lon_name)
    da = smrz_ensure_spatial_dims(da)
    da = da.astype('float32')
    return da


def smrz_prepare_smrz_data(smrz_dir, soc):
    smrz = smrz_open_smrz_dataset(smrz_dir)
    smrz = smrz_ensure_spatial_dims(smrz)
    smrz = smrz.rio.clip(soc.geometry, soc.crs, drop=True).load()
    smrz = smrz_ensure_spatial_dims(smrz)
    if 'time' not in smrz.dims:
        raise ValueError('SMrz data must have a time dimension.')
    time_idx = smrz.indexes['time']
    if hasattr(time_idx, 'to_datetimeindex'):
        time_index = time_idx.to_datetimeindex()
    else:
        time_index = pd.to_datetime(time_idx)
    smrz = smrz.assign_coords(ym=('time', time_index.strftime('%Y-%m')))
    return smrz


def smrz_add_city_labels(ax):
    city_df = pd.DataFrame([(name, lon, lat) for name, (lon, lat)
                           in smrz_CITIES.items()], columns=['City', 'lon', 'lat'])
    city_gdf = gpd.GeoDataFrame(city_df, geometry=gpd.points_from_xy(
        city_df['lon'], city_df['lat']), crs='EPSG:4326').to_crs(epsg=3857)
    for _, row in city_gdf.iterrows():
        ax.plot(row.geometry.x, row.geometry.y, 'ko', markersize=3, zorder=5)
        ax.text(row.geometry.x + 6000, row.geometry.y + 6000, row['City'], fontsize=8, color='black', bbox=dict(
            facecolor='white', edgecolor='none', alpha=0.75, pad=1.2), zorder=6)


def smrz_add_study_area_outline(ax, soc):
    soc_3857 = soc.to_crs(epsg=3857)
    soc_3857.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=4)


def smrz_get_legend_for_sd_mode(meta):
    sd = meta['sd']
    return ([Patch(color='#1a9641', label=f'> +1 SD ({sd:.4f})'), Patch(color='#a6d96a', label='0 to +1 SD'), Patch(color='#d9d9d9', label='Not significant'), Patch(color='#fdae61', label='-1 SD to 0'), Patch(color='#d7191c', label=f'< -1 SD (-{sd:.4f})')], smrz_cmap_5, 0, 4)


def smrz_save_single_map(class_map, soc, out_file, title_text, meta):
    da = class_map.copy()
    da = smrz_ensure_spatial_dims(da)

    # Keep the map in native lon/lat-style geometry rather than Web Mercator.
    # This matches the visual proportions of the correlation/lag figures and
    # prevents the delta maps from looking compressed into square panels.
    da_plot = mask_delta_raster_to_county_land(da, soc)

    legend_handles, use_cmap, vmin, vmax = smrz_get_legend_for_sd_mode(meta)

    fig, ax = plt.subplots(1, 1, figsize=INDIVIDUAL_MAP_FIGSIZE)

    add_county_land_fill(ax, soc, da_plot.rio.crs)

    plot_delta_class_raster(
        ax=ax,
        da=da_plot,
        cmap=use_cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )

    set_map_extent_to_study(ax, soc, da_plot.rio.crs)
    add_county_boundaries_and_study_outline(ax, soc, da_plot.rio.crs)

    # No satellite basemap and no city labels; this matches the lag/correlation map style.
    ax.set_title(title_text, fontsize=14)
    ax.set_axis_off()

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.0, -0.08),
        ncol=2,
        fontsize=10,
    )

    plt.subplots_adjust(bottom=0.17, top=0.88)
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved map: {out_file}')


def smrz_compute_event_deltas(smrz, df_events):
    deltas = []
    for row_i, e in df_events.iterrows():
        s = smrz.sel(time=smrz['ym'].isin(smrz_expand_months(e['StartDate'])))
        f = smrz.sel(time=smrz['ym'].isin(smrz_expand_months(e['EndDate'])))
        if s.sizes.get('time', 0) == 0 or f.sizes.get('time', 0) == 0:
            continue
        d = f.mean('time') - s.mean('time')
        d = d.reset_coords(drop=True)
        event_id = f"K{int(e['K'])}_{str(e['Direction']).replace('->', 'to')}_{pd.to_datetime(e['StartDate']).strftime('%Y%m')}_{pd.to_datetime(e['EndDate']).strftime('%Y%m')}_{row_i}"
        deltas.append({'event_id': event_id, 'transition': str(e['Direction']).replace('->', '_to_'), 'K': int(
            e['K']), 'start_date': pd.to_datetime(e['StartDate']), 'end_date': pd.to_datetime(e['EndDate']), 'delta': d})
    return deltas


def smrz_classify_sd_map(center_map, sig_mask, land_mask):
    vals = center_map.where(sig_mask & land_mask).values
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (xr.full_like(center_map, np.nan), None)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return (xr.full_like(center_map, np.nan), None)
    out = xr.full_like(center_map, np.nan)
    out = out.where(~(center_map < -sd), 0)
    out = out.where(~((center_map >= -sd) & (center_map < 0)), 1)
    out = out.where(~(center_map == 0), 2)
    out = out.where(~((center_map > 0) & (center_map <= sd)), 3)
    out = out.where(~(center_map > sd), 4)
    out = out.where(sig_mask)
    out = out.where(land_mask)
    meta = {'sd': float(sd)}
    return (out, meta)


def smrz_response_class_array(delta, sig_mask, sd):
    out = np.full(delta.shape, 9, dtype=np.int16)
    valid = np.isfinite(delta) & sig_mask
    out[valid & (delta < -sd)] = 0
    out[valid & (delta >= -sd) & (delta < 0)] = 1
    out[valid & (delta == 0)] = 2
    out[valid & (delta > 0) & (delta <= sd)] = 3
    out[valid & (delta > sd)] = 4
    return out


def smrz_load_habitat_polygons():
    pieces = []
    for habitat, shp in smrz_VEG_SHPS.items():
        shp_path = Path(shp)
        if not shp_path.exists():
            warnings.warn(f'Missing habitat shapefile: {shp_path}')
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError(f'Habitat shapefile has no CRS: {shp_path}')
        gdf = smrz_clean_geometries(gdf)
        gdf['habitat'] = habitat
        pieces.append(gdf[['habitat', 'geometry']])
    if len(pieces) == 0:
        raise ValueError('No habitat shapefiles were loaded.')
    habitats = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True), crs=pieces[0].crs)
    habitats = smrz_clean_geometries(habitats)
    return habitats


def smrz_load_elevation_polygons():
    elev_path = Path(smrz_ELEV_SHP)
    if not elev_path.exists():
        fallback = Path(smrz_ELEV_SHP_FALLBACK)
        if not fallback.exists():
            raise FileNotFoundError(
                f'Elevation shapefile was not found:\n{smrz_ELEV_SHP}\n{smrz_ELEV_SHP_FALLBACK}')
        elev_path = fallback
        print(f'Using fallback elevation shapefile: {elev_path}')
    elev = gpd.read_file(elev_path)
    if elev.crs is None:
        raise ValueError(f'Elevation shapefile has no CRS: {elev_path}')
    elev = smrz_clean_geometries(elev)
    bin_field = smrz_choose_elevation_bin_field(elev)
    print(f'Using elevation-bin field: {bin_field}')
    elev['elev_bin_raw'] = elev[bin_field].astype(str)
    elev['elev_bin'] = elev['elev_bin_raw'].map(smrz_make_elevation_label)
    if smrz_ELEV_MIDPOINT_FIELD is not None:
        if smrz_ELEV_MIDPOINT_FIELD not in elev.columns:
            raise ValueError(
                f"ELEV_MIDPOINT_FIELD '{smrz_ELEV_MIDPOINT_FIELD}' was not found.\nAvailable fields: {list(elev.columns)}")
        elev['elev_mid_m'] = pd.to_numeric(
            elev[smrz_ELEV_MIDPOINT_FIELD], errors='coerce')
    else:
        elev['elev_mid_m'] = elev['elev_bin_raw'].map(
            smrz_parse_midpoint_from_bin_label)
    elev['elev_bin'] = elev['elev_bin'].map(smrz_normalize_elevation_bin_label)
    return elev[['elev_bin', 'elev_mid_m', 'geometry']]


def smrz_create_pixel_area_grid_km2(template_da):
    template_da = smrz_ensure_spatial_dims(template_da)
    crs = template_da.rio.crs
    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape
    if crs is not None and crs.is_projected:
        pixel_area = abs(transform.a * transform.e) / 1000000.0
        return np.full((height, width), pixel_area, dtype=float)
    geod = Geod(ellps='WGS84')
    areas = np.full((height, width), np.nan, dtype=float)
    for r in range(height):
        for c in range(width):
            x_ul, y_ul = xy(transform, r, c, offset='ul')
            x_ur, y_ur = xy(transform, r, c, offset='ur')
            x_lr, y_lr = xy(transform, r, c, offset='lr')
            x_ll, y_ll = xy(transform, r, c, offset='ll')
            area_m2, _ = geod.polygon_area_perimeter(
                [x_ul, x_ur, x_lr, x_ll], [y_ul, y_ur, y_lr, y_ll])
            areas[r, c] = abs(area_m2) / 1000000.0
    return areas


def smrz_make_pixel_center_table(template_da):
    template_da = smrz_ensure_spatial_dims(template_da)
    transform = template_da.rio.transform(recalc=True)
    height, width = template_da.shape
    records = []
    for r in range(height):
        for c in range(width):
            lon, lat = xy(transform, r, c, offset='center')
            records.append({'row': int(r), 'col': int(
                c), 'lon': float(lon), 'lat': float(lat)})
    pixels = pd.DataFrame.from_records(records)
    pixel_gdf = gpd.GeoDataFrame(pixels, geometry=gpd.points_from_xy(
        pixels['lon'], pixels['lat']), crs=template_da.rio.crs)
    return pixel_gdf


def smrz_prepare_static_layers(template_da):
    print('\nPreparing static habitat and elevation layers using SMrz pixel-center joins...')
    template_da = smrz_ensure_spatial_dims(template_da)
    habitat_gdf = smrz_load_habitat_polygons()
    elev_gdf = smrz_load_elevation_polygons()
    pixel_points = smrz_make_pixel_center_table(template_da)
    habitat_join = gpd.sjoin(pixel_points, habitat_gdf.to_crs(
        pixel_points.crs), how='left', predicate='within')
    habitat_join = habitat_join.sort_values(
        ['row', 'col']).drop_duplicates(['row', 'col'])
    elev_join = gpd.sjoin(pixel_points, elev_gdf.to_crs(
        pixel_points.crs), how='left', predicate='within')
    elev_join = elev_join.sort_values(
        ['row', 'col', 'elev_mid_m']).drop_duplicates(['row', 'col'])
    static = pixel_points[['row', 'col', 'lon', 'lat', 'geometry']].copy()
    static = static.merge(habitat_join[['row', 'col', 'habitat']], on=[
                          'row', 'col'], how='left')
    static = static.merge(elev_join[['row', 'col', 'elev_bin', 'elev_mid_m']], on=[
                          'row', 'col'], how='left')
    area_km2 = smrz_create_pixel_area_grid_km2(template_da)
    area_records = []
    for r in range(area_km2.shape[0]):
        for c in range(area_km2.shape[1]):
            area_records.append({'row': int(r), 'col': int(
                c), 'area_km2': float(area_km2[r, c])})
    area_df = pd.DataFrame(area_records)
    static = static.merge(area_df, on=['row', 'col'], how='left')
    habitat_order = [
        h for h in smrz_HABITAT_ORDER if h in static['habitat'].dropna().unique()]
    elev_unique = static[['elev_bin', 'elev_mid_m']].dropna(
        subset=['elev_bin']).drop_duplicates().sort_values('elev_mid_m')
    elevation_order = elev_unique['elev_bin'].tolist()
    print('\nHabitat counts assigned to SMrz pixel centers:')
    print(static['habitat'].value_counts(dropna=False))
    print('\nElevation bins assigned to SMrz pixel centers:')
    print(static['elev_bin'].value_counts(dropna=False).sort_index())
    return {'static_pixel_table': static.drop(columns='geometry'), 'habitat_order': habitat_order, 'elevation_order': elevation_order}


def smrz_make_aggregate_pixel_dataframe(center_map, p_map, land_mask, sd_meta, static_layers, transition_label, metric, group_label):
    delta = center_map.values.astype(float)
    pvals = p_map.values.astype(float)
    land = land_mask.values.astype(bool)
    significant = (pvals < smrz_ALPHA) & land & np.isfinite(delta)
    sd = sd_meta['sd']
    response_codes = smrz_response_class_array(delta, significant, sd)
    rows, cols = np.where(land & np.isfinite(delta))
    records = []
    for r, c in zip(rows, cols):
        resp_code = int(response_codes[r, c])
        records.append({'metric': metric, 'variable': smrz_VARIABLE_NAME, 'group': group_label, 'transition': transition_label, 'row': int(r), 'col': int(c), 'delta_smrz': float(delta[r, c]), 'p_value': float(
            pvals[r, c]) if np.isfinite(pvals[r, c]) else np.nan, 'significant': bool(significant[r, c]), 'response_code': resp_code, 'response_class': smrz_RESPONSE_LABELS.get(resp_code, 'Unknown'), 'sd_threshold': float(sd)})
    pixel_df = pd.DataFrame.from_records(records)
    static_df = static_layers['static_pixel_table']
    pixel_df = pixel_df.merge(static_df, on=['row', 'col'], how='left')
    pixel_df['elev_bin'] = pixel_df['elev_bin'].map(
        smrz_normalize_elevation_bin_label)
    pixel_df = pixel_df.loc[pixel_df['habitat'].notna(
    ) & pixel_df['elev_bin'].notna() & np.isfinite(pixel_df['area_km2'])].copy()
    pixel_df['response_direction'] = 'Not significant'
    pixel_df.loc[pixel_df['significant'] & (
        pixel_df['delta_smrz'] < 0), 'response_direction'] = 'Any significant decrease'
    pixel_df.loc[pixel_df['significant'] & (
        pixel_df['delta_smrz'] > 0), 'response_direction'] = 'Any significant increase'
    return pixel_df


def smrz_build_event_level_summary(event_deltas, land_mask, static_layers):
    static_df = static_layers['static_pixel_table'].copy()
    land = land_mask.values.astype(bool)
    rows, cols = np.where(land)
    land_df = pd.DataFrame({'row': rows.astype(int), 'col': cols.astype(int)})
    static_df = land_df.merge(static_df, on=['row', 'col'], how='left')
    static_df = static_df.loc[static_df['habitat'].notna() & static_df['elev_bin'].notna(
    ) & np.isfinite(static_df['area_km2']) & (static_df['area_km2'] > 0)].copy()
    all_records = []
    for ev in event_deltas:
        delta = ev['delta'].values.astype(float)
        values = []
        for _, row in static_df.iterrows():
            r = int(row['row'])
            c = int(row['col'])
            v = delta[r, c]
            if not np.isfinite(v):
                continue
            values.append({'event_id': ev['event_id'], 'transition': ev['transition'], 'K': ev['K'], 'start_date': ev['start_date'], 'end_date': ev['end_date'], 'habitat': row['habitat'],
                          'elev_bin': smrz_normalize_elevation_bin_label(row['elev_bin']), 'elev_mid_m': row['elev_mid_m'], 'delta_smrz': float(v), 'area_km2': float(row['area_km2'])})
        ev_df = pd.DataFrame(values)
        if ev_df.empty:
            continue
        grouped = ev_df.groupby(['event_id', 'transition', 'K', 'start_date', 'end_date', 'habitat', 'elev_bin'], dropna=False).apply(lambda g: pd.Series({'n_pixels': len(g), 'group_area_km2': g['area_km2'].sum(
        ), 'mean_delta_smrz': smrz_weighted_mean(g['delta_smrz'], g['area_km2']), 'median_delta_smrz': float(np.nanmedian(g['delta_smrz'])), 'mean_elev_m': smrz_weighted_mean(g['elev_mid_m'], g['area_km2'])})).reset_index()
        all_records.append(grouped)
    if len(all_records) == 0:
        return pd.DataFrame()
    out = pd.concat(all_records, ignore_index=True)
    out = out.loc[out['habitat'].notna() & out['elev_bin'].notna() & np.isfinite(
        out['mean_delta_smrz']) & np.isfinite(out['mean_elev_m'])].copy()
    return out


def smrz_summarize_response_direction_by_habitat(pixel_df):
    df = pixel_df.copy()
    group_area = df.groupby(['transition', 'habitat'], dropna=False)[
        'area_km2'].sum().reset_index(name='group_area_km2')
    affected = df.loc[df['response_direction'] != 'Not significant'].groupby(
        ['transition', 'habitat', 'response_direction'], dropna=False)['area_km2'].sum().reset_index(name='affected_area_km2')
    out = affected.merge(group_area, on=['transition', 'habitat'], how='left')
    out['percent_of_group_area'] = 100.0 * \
        out['affected_area_km2'] / out['group_area_km2']
    return out


def smrz_summarize_response_composition_by_habitat(pixel_df):
    df = pixel_df.copy()
    group_area = df.groupby(['transition', 'habitat'], dropna=False)[
        'area_km2'].sum().reset_index(name='group_area_km2')
    class_area = df.groupby(['transition', 'habitat', 'response_class'], dropna=False)[
        'area_km2'].sum().reset_index(name='class_area_km2')
    out = class_area.merge(
        group_area, on=['transition', 'habitat'], how='left')
    out['percent_of_group_area'] = 100.0 * \
        out['class_area_km2'] / out['group_area_km2']
    return out


def smrz_save_mirrored_bar_plot_by_habitat(direction_summary, out_png, habitat_order):
    df = direction_summary.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry→Wet', 'Wet_to_Dry': 'Wet→Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        13, max(5, len(habitat_order) * 0.55)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        plot_df = pd.DataFrame({'habitat': habitat_order})
        dec = sub.loc[sub['response_direction'] == 'Any significant decrease', [
            'habitat', 'percent_of_group_area']].rename(columns={'percent_of_group_area': 'decrease_percent'})
        inc = sub.loc[sub['response_direction'] == 'Any significant increase', [
            'habitat', 'percent_of_group_area']].rename(columns={'percent_of_group_area': 'increase_percent'})
        plot_df = plot_df.merge(dec, on='habitat', how='left')
        plot_df = plot_df.merge(inc, on='habitat', how='left')
        plot_df['decrease_percent'] = plot_df['decrease_percent'].fillna(0)
        plot_df['increase_percent'] = plot_df['increase_percent'].fillna(0)
        y = np.arange(len(plot_df))
        ax.barh(y, -plot_df['decrease_percent'], label='Any significant decrease',
                color=smrz_DIRECTION_COLORS['Any significant decrease'], alpha=0.85)
        ax.barh(y, plot_df['increase_percent'], label='Any significant increase',
                color=smrz_DIRECTION_COLORS['Any significant increase'], alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel('Percent of habitat area')
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df['habitat'])
        max_val = max(plot_df['decrease_percent'].max(),
                      plot_df['increase_percent'].max(), 1)
        xlim = max(10, np.ceil(max_val / 10) * 10)
        ax.set_xlim(-xlim, xlim)
        xticks = ax.get_xticks()
        ax.set_xticklabels([f'{abs(x):.0f}' for x in xticks])
        for i, row in plot_df.iterrows():
            if row['decrease_percent'] > 0:
                ax.text(-row['decrease_percent'] - 0.5, i,
                        f"{row['decrease_percent']:.1f}%", va='center', ha='right', fontsize=8)
            if row['increase_percent'] > 0:
                ax.text(row['increase_percent'] + 0.5, i,
                        f"{row['increase_percent']:.1f}%", va='center', ha='left', fontsize=8)
    axes[0].set_ylabel('Habitat')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Significant ΔSMrz response footprint by habitat\nMirrored bars show percent of habitat area with significant decrease or increase', fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved mirrored bar plot: {out_png}')


def smrz_save_stacked_response_composition_bars(composition_df, out_png, habitat_order):
    df = composition_df.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry→Wet', 'Wet_to_Dry': 'Wet→Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        15, max(5, len(habitat_order) * 0.55)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        pivot = sub.pivot_table(index='habitat', columns='response_class',
                                values='percent_of_group_area', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(index=habitat_order,
                              columns=smrz_RESPONSE_CLASS_ORDER, fill_value=0)
        y = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for response_class in smrz_RESPONSE_CLASS_ORDER:
            vals = pivot[response_class].values
            color = smrz_RESPONSE_COLORS.get(response_class, '#cccccc')
            ax.barh(y, vals, left=left, color=color,
                    label=response_class, edgecolor='white', linewidth=0.4)
            left += vals
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel('Percent of habitat area')
        ax.set_xlim(0, 100)
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
    axes[0].set_ylabel('Habitat')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(
        'Habitat response composition during whiplash events\nBars match the ΔSMrz map classes', fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.15, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved stacked composition bars: {out_png}')


def smrz_p_to_stars(p):
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def smrz_extract_habitat_from_term(term):
    m = re.search('C\\(habitat.*?\\)\\[T\\.(.*?)\\]', term)
    if m:
        return m.group(1)
    return None


def smrz_make_significance_stars_from_m6(coef_table):
    stars = {'Dry_to_Wet': {}, 'Wet_to_Dry': {}}
    for _, row in coef_table.iterrows():
        term = str(row['term'])
        p = float(row['p_value'])
        habitat = smrz_extract_habitat_from_term(term)
        if habitat is None:
            continue
        star = smrz_p_to_stars(p)
        if star == '':
            continue
        has_transition = 'C(transition' in term
        has_interaction = ':' in term
        if has_transition and has_interaction:
            stars['Wet_to_Dry'][habitat] = star
        elif not has_transition:
            stars['Dry_to_Wet'][habitat] = star
    return stars


def smrz_save_stacked_response_composition_bars_with_model_stars(composition_df, out_png, habitat_order, significance_stars):
    df = composition_df.copy()
    transitions = ['Dry_to_Wet', 'Wet_to_Dry']
    transition_titles = {'Dry_to_Wet': 'Dry→Wet', 'Wet_to_Dry': 'Wet→Dry'}
    fig, axes = plt.subplots(1, 2, figsize=(
        16, max(5.5, len(habitat_order) * 0.58)), sharey=True)
    for ax, transition in zip(axes, transitions):
        sub = df.loc[df['transition'] == transition].copy()
        pivot = sub.pivot_table(index='habitat', columns='response_class',
                                values='percent_of_group_area', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(index=habitat_order,
                              columns=smrz_RESPONSE_CLASS_ORDER, fill_value=0)
        y = np.arange(len(pivot))
        left = np.zeros(len(pivot))
        for response_class in smrz_RESPONSE_CLASS_ORDER:
            vals = pivot[response_class].values
            color = smrz_RESPONSE_COLORS.get(response_class, '#cccccc')
            ax.barh(y, vals, left=left, color=color,
                    label=response_class, edgecolor='white', linewidth=0.4)
            left += vals
        ax.set_title(transition_titles.get(
            transition, transition), fontsize=13)
        ax.set_xlabel('Percent of habitat area')
        ax.set_xlim(0, 108)
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index)
        stars_for_transition = significance_stars.get(transition, {})
        for i, habitat in enumerate(pivot.index):
            stars = stars_for_transition.get(habitat, '')
            if stars:
                ax.text(102.0, i, stars, va='center', ha='left',
                        fontsize=12, fontweight='bold', color='black')
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel('Habitat')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle('Habitat response composition during whiplash events\nBars match the ΔSMrz map classes, stars show event-level model significance', fontsize=15, y=0.98)
    note = 'Asterisks indicate significant event-level SMrz model terms relative to Shrub: * p < 0.05, ** p < 0.01, *** p < 0.001. Dry→Wet markers show habitat main effects; Wet→Dry markers show transition × habitat interactions.'
    fig.text(0.5, 0.025, note, ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, 0.17, 1, 0.92])
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved stacked composition bars with model stars: {out_png}')


def smrz_save_model_results(model, out_txt, title):
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write('=' * len(title) + '\n\n')
        f.write(str(model.summary()))


def smrz_model_to_coef_table(model):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    bse = model.bse
    out = pd.DataFrame({'term': params.index, 'estimate': params.values, 'std_error': bse.values,
                       'p_value': pvals.values, 'ci_lower': conf[0].values, 'ci_upper': conf[1].values})
    return out


def smrz_fit_wls_cluster(formula, df, weight_col='group_area_km2', cluster_col='event_id'):
    model = smf.wls(formula=formula, data=df, weights=df[weight_col]).fit(
        cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
    return model


def smrz_run_event_level_statistics(event_summary_df, out_dir):
    stats_dir = Path(out_dir) / 'Event_Level_Statistics'
    stats_dir.mkdir(parents=True, exist_ok=True)
    df = event_summary_df.copy()
    df = df.loc[df['habitat'].notna() & df['transition'].notna() & np.isfinite(df['mean_delta_smrz']) & np.isfinite(
        df['mean_elev_m']) & np.isfinite(df['group_area_km2']) & (df['group_area_km2'] > 0)].copy()
    df['transition'] = pd.Categorical(df['transition'], categories=[
                                      'Dry_to_Wet', 'Wet_to_Dry'], ordered=False)
    habitat_categories = ['Shrub', 'Agriculture', 'Conifer_Forest',
                          'Conifer_Woodland', 'Desert', 'Hardwood', 'Herbaceous', 'Urban']
    habitat_categories = [
        h for h in habitat_categories if h in df['habitat'].dropna().unique()]
    if smrz_REFERENCE_HABITAT not in habitat_categories:
        raise ValueError(
            f"Reference habitat '{smrz_REFERENCE_HABITAT}' not found in model data. Available habitats are: {habitat_categories}")
    df['habitat'] = pd.Categorical(
        df['habitat'], categories=habitat_categories, ordered=False)
    df.to_csv(stats_dir / 'event_level_model_input.csv', index=False)
    model_specs = {'M1_transition_only': "mean_delta_smrz ~ C(transition, Treatment(reference='Dry_to_Wet'))", 'M2_habitat_only': "mean_delta_smrz ~ C(habitat, Treatment(reference='Shrub'))", 'M3_elevation_only': 'mean_delta_smrz ~ mean_elev_m', 'M4_transition_habitat': "mean_delta_smrz ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(habitat, Treatment(reference='Shrub'))",
                   'M5_transition_elevation': "mean_delta_smrz ~ C(transition, Treatment(reference='Dry_to_Wet')) * mean_elev_m", 'M6_full_transition_habitat_elevation': "mean_delta_smrz ~ C(transition, Treatment(reference='Dry_to_Wet')) * C(habitat, Treatment(reference='Shrub')) + C(transition, Treatment(reference='Dry_to_Wet')) * mean_elev_m"}
    model_summaries = []
    m6_coef_table = None
    for name, formula in model_specs.items():
        try:
            model = smrz_fit_wls_cluster(formula, df)
            smrz_save_model_results(
                model, stats_dir / f'{name}_SMRZ_SHRUB_REFERENCE_cluster_robust_summary.txt', title=f'{name}: {formula}')
            coef_table = smrz_model_to_coef_table(model)
            coef_table.insert(0, 'variable', smrz_VARIABLE_NAME)
            coef_table.insert(1, 'reference_transition',
                              smrz_REFERENCE_TRANSITION)
            coef_table.insert(2, 'reference_habitat', smrz_REFERENCE_HABITAT)
            coef_table.to_csv(
                stats_dir / f'{name}_SMRZ_SHRUB_REFERENCE_cluster_robust_coefficients.csv', index=False)
            if name == 'M6_full_transition_habitat_elevation':
                m6_coef_table = coef_table.copy()
            model_summaries.append({'variable': smrz_VARIABLE_NAME, 'model': name, 'formula': formula, 'reference_transition': smrz_REFERENCE_TRANSITION, 'reference_habitat': smrz_REFERENCE_HABITAT, 'n_rows': int(
                model.nobs), 'r_squared': getattr(model, 'rsquared', np.nan), 'aic': getattr(model, 'aic', np.nan), 'bic': getattr(model, 'bic', np.nan)})
        except Exception as e:
            model_summaries.append({'variable': smrz_VARIABLE_NAME, 'model': name, 'formula': formula,
                                   'reference_transition': smrz_REFERENCE_TRANSITION, 'reference_habitat': smrz_REFERENCE_HABITAT, 'error': str(e)})
    pd.DataFrame(model_summaries).to_csv(
        stats_dir / 'model_summary_table_SMRZ_SHRUB_REFERENCE.csv', index=False)
    kruskal_records = []
    try:
        hab_groups = [g['mean_delta_smrz'].dropna().values for _, g in df.groupby(
            'habitat', observed=True) if len(g['mean_delta_smrz'].dropna()) >= 2]
        if len(hab_groups) >= 2:
            H, p = kruskal(*hab_groups)
            kruskal_records.append({'variable': smrz_VARIABLE_NAME, 'test': 'Kruskal-Wallis habitat difference', 'statistic': H,
                                   'p_value': p, 'reference_habitat': smrz_REFERENCE_HABITAT, 'note': 'Uses event-level habitat-elevation summaries.'})
    except Exception as e:
        kruskal_records.append({'variable': smrz_VARIABLE_NAME, 'test': 'Kruskal-Wallis habitat difference',
                               'reference_habitat': smrz_REFERENCE_HABITAT, 'error': str(e)})
    try:
        trans_groups = [g['mean_delta_smrz'].dropna().values for _, g in df.groupby(
            'transition', observed=True) if len(g['mean_delta_smrz'].dropna()) >= 2]
        if len(trans_groups) == 2:
            H, p = kruskal(*trans_groups)
            kruskal_records.append({'variable': smrz_VARIABLE_NAME, 'test': 'Kruskal-Wallis transition difference', 'statistic': H, 'p_value': p,
                                   'reference_transition': smrz_REFERENCE_TRANSITION, 'note': 'Tests Dry_to_Wet vs Wet_to_Dry using event-level summaries.'})
    except Exception as e:
        kruskal_records.append({'variable': smrz_VARIABLE_NAME, 'test': 'Kruskal-Wallis transition difference',
                               'reference_transition': smrz_REFERENCE_TRANSITION, 'error': str(e)})
    pd.DataFrame(kruskal_records).to_csv(
        stats_dir / 'nonparametric_tests_SMRZ_SHRUB_REFERENCE.csv', index=False)
    if m6_coef_table is not None:
        significance_stars = smrz_make_significance_stars_from_m6(
            m6_coef_table)
        with open(stats_dir / 'M6_SMRZ_significance_stars.json', 'w') as f:
            json.dump(significance_stars, f, indent=4)
    else:
        significance_stars = {'Dry_to_Wet': {}, 'Wet_to_Dry': {}}
    print(f'Saved event-level statistics to: {stats_dir}')
    print('Reference habitat for model coefficients: Shrub')
    print('Reference transition for model coefficients: Dry_to_Wet')
    return significance_stars


def smrz_save_rasters(center_map, p_map, out_dir, out_name_base):
    if not smrz_SAVE_RASTERS:
        return
    raster_dir = Path(out_dir) / 'GeoTIFFs'
    raster_dir.mkdir(parents=True, exist_ok=True)
    center_map = smrz_ensure_spatial_dims(center_map)
    p_map = smrz_ensure_spatial_dims(p_map)
    center_map.rio.to_raster(
        raster_dir / f'{out_name_base}_center_delta_smrz.tif')
    p_map.rio.to_raster(raster_dir / f'{out_name_base}_wilcoxon_p.tif')
    print(f'Saved GeoTIFFs to: {raster_dir}')


def smrz_process_direction(smrz, soc, land_mask, df_events, direction, out_dir, out_name_base, title_text_base, static_layers, group_label):
    if len(df_events) < smrz_MIN_EVENTS:
        print(f'Skipping {direction}, not enough events: {len(df_events)}')
        return (None, None)
    event_deltas = smrz_compute_event_deltas(smrz, df_events)
    if len(event_deltas) < smrz_MIN_EVENTS:
        print(
            f'Skipping {direction}, not enough valid deltas: {len(event_deltas)}')
        return (None, None)
    stack = xr.concat([ev['delta'] for ev in event_deltas],
                      dim='event', coords='minimal', compat='override')
    center_map, p_map = xr.apply_ufunc(lambda x: smrz_run_wilcoxon(x), stack, input_core_dims=[
                                       ['event']], output_core_dims=[[], []], vectorize=True, output_dtypes=[float, float])
    center_map = center_map.rio.write_crs(smrz.rio.crs)
    p_map = p_map.rio.write_crs(smrz.rio.crs)
    center_map = smrz_ensure_spatial_dims(center_map)
    p_map = smrz_ensure_spatial_dims(p_map)
    sig_mask = (p_map < smrz_ALPHA) & land_mask
    out_dir.mkdir(parents=True, exist_ok=True)
    smrz_save_rasters(center_map, p_map, out_dir, out_name_base)
    sd_map, sd_meta = smrz_classify_sd_map(center_map, sig_mask, land_mask)
    if sd_meta is None:
        print(f'No valid SD map for {direction}')
        return (None, None)
    smrz_save_single_map(class_map=sd_map, soc=soc, out_file=out_dir / f'{out_name_base}_PlusMinus1SD.png',
                         title_text=f'{title_text_base}\nSignificant Pixels Only (Wilcoxon p < {smrz_ALPHA}) | Data-driven bins, ±1 SD', meta=sd_meta)
    transition_label = direction.replace('->', '_to_')
    aggregate_pixel_df = smrz_make_aggregate_pixel_dataframe(center_map=center_map, p_map=p_map, land_mask=land_mask, sd_meta=sd_meta,
                                                             static_layers=static_layers, transition_label=transition_label, metric=smrz_METRIC, group_label=group_label)
    event_summary_df = smrz_build_event_level_summary(
        event_deltas=event_deltas, land_mask=land_mask, static_layers=static_layers)
    transition_dir = Path(out_dir) / f'Habitat_Stats_{transition_label}'
    transition_dir.mkdir(parents=True, exist_ok=True)
    aggregate_pixel_df.to_csv(
        transition_dir / 'aggregate_pixel_table_for_figures.csv', index=False)
    event_summary_df.to_csv(
        transition_dir / 'event_level_habitat_elevation_summary.csv', index=False)
    return (aggregate_pixel_df, event_summary_df)


def smrz_prepare_smrz_data(smrz_dir, soc):
    """
    Load, clip, and restrict SMrz to January 1982-December 2022.

    This definition intentionally replaces the earlier helper while preserving
    the remainder of the original SMrz analysis.
    """
    smrz = smrz_open_smrz_dataset(smrz_dir)
    smrz = smrz_ensure_spatial_dims(smrz)
    smrz = smrz.rio.clip(soc.geometry, soc.crs, drop=True)
    smrz = smrz_ensure_spatial_dims(smrz)
    if 'time' not in smrz.dims:
        raise ValueError('SMrz data must have a time dimension.')
    time_idx = smrz.indexes['time']
    if hasattr(time_idx, 'to_datetimeindex'):
        time_index = time_idx.to_datetimeindex()
    else:
        time_index = pd.DatetimeIndex(pd.to_datetime(time_idx))
    keep = (time_index >= smrz_ANALYSIS_START) & (
        time_index <= smrz_ANALYSIS_END)
    if not np.any(keep):
        raise ValueError(
            'SMrz contains no observations from January 1982 through December 2022.')
    smrz = smrz.isel(time=np.flatnonzero(keep)).load()
    time_index = pd.DatetimeIndex(time_index[keep])
    smrz = smrz.assign_coords(ym=('time', time_index.strftime('%Y-%m')))
    print(
        f'SMrz analysis period: {time_index.min():%Y-%m} to {time_index.max():%Y-%m}; {len(time_index):,} time steps.')
    return smrz


def smrz_prepare_smrz_common_data():
    print('\nLoading common SMrz data for all three detection methods...')
    soc = gpd.read_file(smrz_SOCAL_SHP)
    smrz = smrz_prepare_smrz_data(smrz_SMRZ_DIR, soc)
    land_mask = smrz.notnull().any('time')
    land_mask = land_mask.rio.write_crs(smrz.rio.crs)
    land_mask = smrz_ensure_spatial_dims(land_mask)
    static_layers = smrz_prepare_static_layers(smrz.isel(time=0))
    return {'soc': soc, 'smrz': smrz, 'land_mask': land_mask, 'static_layers': static_layers}


def smrz_run_smrz_detection_method(common_data, detection_method):
    print('\n' + '#' * 79)
    print(f'SMrz WHIPLASH DETECTION METHOD: {detection_method}')
    print('#' * 79)
    soc = common_data['soc']
    smrz = common_data['smrz']
    land_mask = common_data['land_mask']
    static_layers = common_data['static_layers']
    df_all_k = smrz_load_whiplash_events(detection_method)
    out_dir = smrz_BASE_OUT_ROOT / detection_method / \
        smrz_VARIABLE_NAME / smrz_METRIC / 'All_K' / 'RW_3-Month_Mean'
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_tables = []
    event_summary_tables = []
    for direction in smrz_DIRECTIONS:
        df_combo = df_all_k[df_all_k['Direction'] == direction].copy()
        if len(df_combo) < smrz_MIN_EVENTS:
            print(f'Skipping {direction}, not enough events: {len(df_combo)}')
            continue
        out_name_base = f"GLEAM_{smrz_VARIABLE_NAME}_{smrz_METRIC}_AllK_{direction.replace('->', 'To')}"
        title_text = f'GLEAM Δ{smrz_VARIABLE_NAME} ({smrz_METRIC}) | {direction}\nAll K Combined, Window=3-Month Mean, Events={len(df_combo)}'
        print(f'\nProcessing {direction}, events={len(df_combo)}')
        agg_df, event_df = smrz_process_direction(smrz=smrz, soc=soc, land_mask=land_mask, df_events=df_combo, direction=direction,
                                                  out_dir=out_dir, out_name_base=out_name_base, title_text_base=title_text, static_layers=static_layers, group_label='All_K')
        if agg_df is not None:
            aggregate_tables.append(agg_df)
        if event_df is not None:
            event_summary_tables.append(event_df)
    significance_stars = {'Dry_to_Wet': {}, 'Wet_to_Dry': {}}
    if len(event_summary_tables) > 0:
        event_all = pd.concat(event_summary_tables, ignore_index=True)
        event_all.to_csv(
            out_dir / 'combined_event_level_habitat_elevation_summary.csv', index=False)
        if smrz_SAVE_STATS:
            significance_stars = smrz_run_event_level_statistics(
                event_all, out_dir)
    if len(aggregate_tables) > 0:
        aggregate_all = pd.concat(aggregate_tables, ignore_index=True)
        figures_dir = out_dir / 'Publishable_Figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        aggregate_all.to_csv(
            figures_dir / 'combined_aggregate_pixel_table_for_figures.csv', index=False)
        direction_summary = smrz_summarize_response_direction_by_habitat(
            aggregate_all)
        composition_summary = smrz_summarize_response_composition_by_habitat(
            aggregate_all)
        direction_summary.to_csv(
            figures_dir / 'summary_any_significant_decrease_increase_by_habitat.csv', index=False)
        composition_summary.to_csv(
            figures_dir / 'summary_response_composition_by_habitat.csv', index=False)
        habitat_order_for_figs = [
            h for h in smrz_HABITAT_ORDER if h in aggregate_all['habitat'].dropna().unique()]
        smrz_save_mirrored_bar_plot_by_habitat(direction_summary=direction_summary, out_png=figures_dir /
                                               'figure_mirrored_bar_any_significant_response_by_habitat.png', habitat_order=habitat_order_for_figs)
        smrz_save_stacked_response_composition_bars(composition_df=composition_summary, out_png=figures_dir /
                                                    'figure_stacked_response_composition_by_habitat.png', habitat_order=habitat_order_for_figs)
        smrz_save_stacked_response_composition_bars_with_model_stars(
            composition_df=composition_summary, out_png=figures_dir / 'figure_stacked_response_composition_by_habitat_WITH_MODEL_STARS.png', habitat_order=habitat_order_for_figs, significance_stars=significance_stars)
    print('\nFinished.')
    print(f'Main output folder:\n{out_dir}')
    print('\nKey new outputs:')
    print('  Publishable_Figures/figure_mirrored_bar_any_significant_response_by_habitat.png')
    print('  Publishable_Figures/figure_stacked_response_composition_by_habitat.png')
    print('  Publishable_Figures/figure_stacked_response_composition_by_habitat_WITH_MODEL_STARS.png')
    print('  Event_Level_Statistics/model_summary_table_SMRZ_SHRUB_REFERENCE.csv')
    print('  Event_Level_Statistics/M6_full_transition_habitat_elevation_SMRZ_SHRUB_REFERENCE_cluster_robust_coefficients.csv')
    print('  Event_Level_Statistics/nonparametric_tests_SMRZ_SHRUB_REFERENCE.csv')
    print('  Event_Level_Statistics/M6_SMRZ_significance_stars.json')


def smrz_main():
    print('\nStarting combined SMrz response analysis for all three whiplash detection methods.')
    print(
        f'Required analysis period: {smrz_ANALYSIS_START:%Y-%m} through {smrz_ANALYSIS_END:%Y-%m}')
    common_data = smrz_prepare_smrz_common_data()
    for detection_method in smrz_DETECTION_METHODS:
        smrz_run_smrz_detection_method(common_data, detection_method)
    print('\nFinished all SMrz detection-method analyses.')


RUN_NDVI = True
RUN_SMRZ = True


def main():
    if RUN_NDVI:
        ndvi_main()

    if RUN_SMRZ:
        smrz_main()

    print("\nCombined standalone NDVI and SMrz analyses are complete.")


if __name__ == "__main__":
    main()
