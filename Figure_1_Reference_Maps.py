# ============================================================

# Figure 1 reference maps:

#

# (a) California inset with study area

# (b) Habitat map with county labels

# (c) Elevation pixel/class map with boxed inset color ramp

#

# Also creates a gridded 3-panel figure:

# (a) Study Area, spanning the full left half

# (b) Habitat Types, upper right

# (c) Elevation Classes, lower right

#

# Updates:

#   - Same font family across the full figure

#   - Same map-label font size across all maps

#   - Larger white elevation legend boxes

#   - Elevation numbers match between panels (c) and (d)

#   - Mountain/location labels are bold and consistent

#   - Capitalized panel titles

#

# Required packages:

#   geopandas, pandas, numpy, matplotlib, rasterio, shapely, pyproj

# ============================================================


import os

import warnings


import numpy as np

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.colors as mcolors


from matplotlib.ticker import FuncFormatter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import rasterio

from rasterio.enums import Resampling

from rasterio.windows import from_bounds

from rasterio.features import geometry_mask

from affine import Affine

from pyproj import Transformer


warnings.filterwarnings("ignore")


# ------------------------------------------------------------

# 0. Global font settings

# ------------------------------------------------------------


BASE_FONT = "Arial"


plt.rcParams.update({

    "font.family": BASE_FONT,

    "font.sans-serif": [BASE_FONT, "DejaVu Sans", "Liberation Sans"],

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "axes.titlesize": 12,

    "axes.labelsize": 9,

    "xtick.labelsize": 8,

    "ytick.labelsize": 8,

})


# ------------------------------------------------------------

# 1. File paths

# ------------------------------------------------------------


COUNTY_SHP = r"D:\Graduate Research\Graduate Research\Study Area Extent\ca_counties\CA_Counties.shp"


BBOX_SHP = r"D:\Graduate Research\Graduate Research\Data\BoundingBox\BoundingBox.shp"


ELEV_SHP = r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Habitats_Elevation\Elevation_Different_Resolutions\elevation_classified_8km_250m_bins_land_northup.shp"


ELEV_SHP_FALLBACK = r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Habitats_Elevation\Elevation_Different_Resolutions\elevation_classified_1km_250m_bins_land_northup.shp"


ELEV_DEM_TIF = r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Habitats_Elevation\Elevation_Different_Resolutions\elevation_30m_land_northup.tif"


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


OUT_DIR = r"D:\Graduate Research\Final_Paper_Analysis\Figure_1"

os.makedirs(OUT_DIR, exist_ok=True)


OUT_INSET_PNG = os.path.join(OUT_DIR, "Figure1a_inset_study_area.png")

OUT_HABITAT_PNG = os.path.join(OUT_DIR, "Figure1b_habitat_map.png")

OUT_ELEV_PIXEL_PNG = os.path.join(OUT_DIR, "Figure1c_elevation_pixel_map.png")

OUT_ELEV_CONTOUR_PNG = os.path.join(

    OUT_DIR, "Figure1d_dem_elevation_contours.png")

OUT_GRIDDED_PNG = os.path.join(OUT_DIR, "Figure1_gridded_reference_maps.png")


OUT_INSET_PDF = os.path.join(OUT_DIR, "Figure1a_inset_study_area.pdf")

OUT_HABITAT_PDF = os.path.join(OUT_DIR, "Figure1b_habitat_map.pdf")

OUT_ELEV_PIXEL_PDF = os.path.join(OUT_DIR, "Figure1c_elevation_pixel_map.pdf")

OUT_ELEV_CONTOUR_PDF = os.path.join(

    OUT_DIR, "Figure1d_dem_elevation_contours.pdf")

OUT_GRIDDED_PDF = os.path.join(OUT_DIR, "Figure1_gridded_reference_maps.pdf")


# ------------------------------------------------------------

# 2. Colors, labels, and map settings

# ------------------------------------------------------------


STUDY_EXTENT_PAD_X = 0.0

STUDY_EXTENT_PAD_Y = 0.0


PANEL_LABEL_FONTSIZE = 13

MAP_LABEL_FONTSIZE = 7.4

MAP_TITLE_FONTSIZE = 12

MAIN_TITLE_FONTSIZE = 15

LEGEND_FONTSIZE = 6.2

LEGEND_TITLE_FONTSIZE = 6.8

TICK_LABEL_FONTSIZE = 8


HABITAT_COLORS = {

    "Agriculture": "#D9D9D9",

    "Conifer_Forest": "#3CB043",

    "Conifer_Woodland": "#1E5631",

    "Desert": "#F6DFA6",

    "Hardwood": "#2C7FB8",

    "Herbaceous": "#8C4F2B",

    "Shrub": "#C98C1A",

    "Urban": "#555555",

}


HABITAT_DISPLAY = {

    "Agriculture": "Agriculture",

    "Conifer_Forest": "Conifer Forest",

    "Conifer_Woodland": "Conifer Woodland",

    "Desert": "Desert",

    "Hardwood": "Hardwood",

    "Herbaceous": "Herbaceous",

    "Shrub": "Shrub",

    "Urban": "Urban",

}


NOT_ANALYZED_COLOR = "#F2F2F2"


ELEV_CMAP = mcolors.LinearSegmentedColormap.from_list(

    "elevation_yellow_orange_purple_white",

    [

        "#FFF2A8",

        "#F3C567",

        "#E99B50",

        "#D46A5C",

        "#B85C73",

        "#8E5C85",

        "#B69AC8",

        "#E6DCEB",

        "#FFFFFF",

    ],

    N=256,

)


DEM_DOWNSAMPLE_FACTOR = 20


DEM_FILLED_CONTOUR_INTERVAL_M = 250

DEM_LINE_CONTOUR_INTERVAL_M = 500

DEM_LABEL_CONTOUR_INTERVAL_M = 1000


DEM_CONTOUR_LINE_COLOR = "#5A4636"

DEM_CONTOUR_LABEL_COLOR = "#5A4636"


COUNTIES_TO_LABEL = [

    "Santa Barbara",

    "Ventura",

    "Los Angeles",

    "Orange",

    "Riverside",

    "San Bernardino",

    "Kern",

    "San Diego",

]


COUNTY_LABEL_POSITIONS = {

    "Santa Barbara": (-119.85, 34.72),

    "Ventura": (-119.10, 34.43),

    "Los Angeles": (-118.32, 34.08),

    "Orange": (-117.75, 33.72),

    "Riverside": (-117.18, 33.88),

    "San Bernardino": (-116.70, 34.45),

    "Kern": (-118.80, 35.05),

    "San Diego": (-116.95, 32.85),

}


GEOGRAPHIC_LABELS = [

    ("Santa Ynez\nMtns.", -119.55, 34.63, 0),

    ("San Gabriel\nMtns.", -117.90, 34.36, 0),

    ("San Bernardino\nMtns.", -116.95, 34.20, 0),

    ("Laguna\nMtns.", -116.45, 32.95, 0),

    ("Channel Islands", -119.35, 33.70, 0),

]


# ------------------------------------------------------------

# 3. Helper functions

# ------------------------------------------------------------


def clean_gdf(gdf):

    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notna()]

    gdf = gdf[~gdf.geometry.is_empty]

    try:

        from shapely.validation import make_valid

        gdf["geometry"] = gdf.geometry.apply(make_valid)

    except Exception:

        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna()]

    gdf = gdf[~gdf.geometry.is_empty]

    return gdf


def read_shp(path, label):

    if not os.path.exists(path):

        raise FileNotFoundError(f"{label} file not found:\n{path}")

    gdf = gpd.read_file(path)

    if gdf.empty:

        raise ValueError(f"{label} file is empty:\n{path}")

    return clean_gdf(gdf)


def to_wgs84(gdf, label):

    gdf = gdf.copy()

    if gdf.crs is None:

        print(f"WARNING: {label} has no CRS. Assuming EPSG:4326.")

        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    else:

        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def union_geometry(gdf):

    try:

        return gdf.geometry.union_all()

    except Exception:

        return gdf.geometry.unary_union


def dissolve_to_one(gdf):

    return gpd.GeoDataFrame(

        geometry=[union_geometry(gdf)],

        crs=gdf.crs,

    )


def safe_clip(gdf, mask, label):

    try:

        clipped = gpd.clip(gdf, mask)

        clipped = clean_gdf(clipped)

        return clipped

    except Exception as e:

        print(

            f"Clip failed for {label}. Returning unclipped layer. Error: {e}")

        return gdf


def find_county_name_col(gdf):

    candidates = [

        "NAME", "Name", "name",

        "COUNTY", "County", "county",

        "COUNTY_NAM", "COUNTY_NAME", "CountyName",

        "NAMELSAD",

    ]

    for col in candidates:

        if col in gdf.columns:

            return col

    raise ValueError(

        "Could not detect county name column. Available columns:\n"

        + "\n".join(gdf.columns.astype(str))

    )


def find_elevation_col(gdf):

    candidates = [

        "elev_bin", "ELEV_BIN", "Elev_Bin",

        "elevation", "ELEVATION", "Elevation",

        "elev", "ELEV", "Elev",

        "gridcode", "GRIDCODE", "Gridcode",

        "VALUE", "Value", "value",

        "class", "CLASS", "Class",

        "DN",

    ]

    for col in candidates:

        if col in gdf.columns:

            return col

    numeric_cols = [

        col for col in gdf.columns

        if col != "geometry" and pd.api.types.is_numeric_dtype(gdf[col])

    ]

    for col in numeric_cols:

        nunique = gdf[col].nunique(dropna=True)

        if 2 <= nunique <= 50:

            return col

    raise ValueError(

        "Could not detect elevation class column. Available columns:\n"

        + "\n".join(gdf.columns.astype(str))

    )


def build_elevation_labels(elev_gdf, elev_col):

    elev_gdf = elev_gdf.copy()

    vals = elev_gdf[elev_col].dropna()

    if len(vals) == 0:

        raise ValueError("Elevation column contains no non-null values.")

    if not pd.api.types.is_numeric_dtype(vals):

        elev_gdf["elev_label"] = elev_gdf[elev_col].astype(str)

        order = sorted(elev_gdf["elev_label"].unique())

        elev_gdf["elev_order"] = elev_gdf["elev_label"].map(

            {lab: i for i, lab in enumerate(order)}

        )

        elev_gdf["elev_lower_m"] = elev_gdf["elev_order"] * 250

        elev_gdf["elev_upper_m"] = elev_gdf["elev_lower_m"] + 250

        elev_gdf["elev_mid_m"] = (

            elev_gdf["elev_lower_m"] + elev_gdf["elev_upper_m"]

        ) / 2.0

        return elev_gdf

    unique_vals = sorted(vals.unique())

    if max(unique_vals) <= 50:

        val_to_order = {v: i for i, v in enumerate(unique_vals)}

        val_to_lower = {v: i * 250 for i, v in enumerate(unique_vals)}

        val_to_upper = {v: (i + 1) * 250 for i, v in enumerate(unique_vals)}

        val_to_label = {

            v: f"{val_to_lower[v]}-{val_to_upper[v]} m"

            for v in unique_vals

        }

    else:

        val_to_order = {}

        val_to_lower = {}

        val_to_upper = {}

        val_to_label = {}

        for i, v in enumerate(unique_vals):

            lower = int(np.floor(float(v) / 250.0) * 250)

            upper = lower + 250

            val_to_order[v] = i

            val_to_lower[v] = lower

            val_to_upper[v] = upper

            val_to_label[v] = f"{lower}-{upper} m"

    elev_gdf["elev_order"] = elev_gdf[elev_col].map(val_to_order)

    elev_gdf["elev_label"] = elev_gdf[elev_col].map(val_to_label)

    elev_gdf["elev_lower_m"] = elev_gdf[elev_col].map(val_to_lower)

    elev_gdf["elev_upper_m"] = elev_gdf[elev_col].map(val_to_upper)

    elev_gdf["elev_mid_m"] = (

        elev_gdf["elev_lower_m"] + elev_gdf["elev_upper_m"]

    ) / 2.0

    return elev_gdf


def lon_formatter(x, pos):

    return f"{abs(x):.1f}°W" if x < 0 else f"{x:.1f}°E"


def lat_formatter(y, pos):

    return f"{abs(y):.1f}°S" if y < 0 else f"{y:.1f}°N"


def format_lonlat_axes(ax, label_size=TICK_LABEL_FONTSIZE):

    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))

    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

    ax.tick_params(labelsize=label_size)

    ax.set_xlabel("")

    ax.set_ylabel("")

    ax.set_aspect("equal", adjustable="box")

    ax.margins(0)


def add_panel_label(ax, label, color="black"):

    ax.text(

        0.015,

        0.985,

        label,

        transform=ax.transAxes,

        ha="left",

        va="top",

        fontsize=PANEL_LABEL_FONTSIZE,

        fontweight="bold",

        color=color,

        bbox=dict(

            facecolor="white",

            edgecolor="none",

            alpha=0.82,

            boxstyle="round,pad=0.16",

        ),

        zorder=50,

    )


def set_study_extent(ax, study_boundary, pad_x=STUDY_EXTENT_PAD_X, pad_y=STUDY_EXTENT_PAD_Y):

    minx, miny, maxx, maxy = study_boundary.total_bounds

    dx = maxx - minx

    dy = maxy - miny

    ax.set_xlim(minx - dx * pad_x, maxx + dx * pad_x)

    ax.set_ylim(miny - dy * pad_y, maxy + dy * pad_y)

    ax.margins(0)


def add_county_labels(ax):

    for county in COUNTIES_TO_LABEL:

        if county in COUNTY_LABEL_POSITIONS:

            x, y = COUNTY_LABEL_POSITIONS[county]

            ax.text(

                x,

                y,

                county,

                fontsize=MAP_LABEL_FONTSIZE,

                ha="center",

                va="center",

                color="black",

                fontweight="bold",

                fontfamily=BASE_FONT,

                bbox=dict(

                    facecolor="white",

                    edgecolor="none",

                    alpha=0.70,

                    boxstyle="round,pad=0.12",

                ),

                zorder=20,

            )


def add_geographic_labels(ax):

    for label, x, y, rotation in GEOGRAPHIC_LABELS:

        ax.text(

            x,

            y,

            label,

            fontsize=MAP_LABEL_FONTSIZE,

            ha="center",

            va="center",

            color="black",

            fontstyle="italic",

            fontweight="bold",

            fontfamily=BASE_FONT,

            rotation=rotation,

            bbox=dict(

                facecolor="white",

                edgecolor="none",

                alpha=0.70,

                boxstyle="round,pad=0.10",

            ),

            zorder=20,

        )


def make_shared_elevation_ticks(elev_min, elev_max, n_ticks=4):

    tick_values = np.linspace(elev_min, elev_max, n_ticks)

    tick_values = np.round(tick_values / 250) * 250

    tick_values = sorted(set(tick_values.astype(int).tolist()))

    tick_labels = [f"{v} m" for v in tick_values]

    return tick_values, tick_labels


def add_boxed_gradient_legend(

    ax,

    cmap,

    norm_for_bar,

    tick_values,

    tick_labels_for_bar,

    title="Elevation",

    width="36%",

    height="39%",

    loc="lower left",

    borderpad=0.78,

):

    legend_ax = inset_axes(

        ax,

        width=width,

        height=height,

        loc=loc,

        borderpad=borderpad,

    )

    legend_ax.set_facecolor("white")

    legend_ax.patch.set_alpha(1.0)

    legend_ax.set_zorder(100)

    for spine in legend_ax.spines.values():

        spine.set_visible(True)

        spine.set_edgecolor("#BDBDBD")

        spine.set_linewidth(0.65)

    vmin = float(norm_for_bar.vmin)

    vmax = float(norm_for_bar.vmax)

    ypad = (vmax - vmin) * 0.10

    legend_ax.set_xlim(0, 1)

    legend_ax.set_ylim(vmin - ypad, vmax + ypad)

    gradient = np.linspace(vmin, vmax, 256).reshape(256, 1)

    legend_ax.imshow(

        gradient,

        extent=(0.10, 0.28, vmin, vmax),

        origin="lower",

        aspect="auto",

        cmap=cmap,

        norm=norm_for_bar,

        zorder=2,

    )

    legend_ax.set_xticks([])

    legend_ax.set_yticks([])

    legend_ax.text(

        0.06,

        0.96,

        title,

        transform=legend_ax.transAxes,

        ha="left",

        va="top",

        fontsize=LEGEND_TITLE_FONTSIZE,

        fontfamily=BASE_FONT,

        color="black",

        zorder=4,

    )

    for tick_value, tick_label in zip(tick_values, tick_labels_for_bar):

        legend_ax.plot(

            [0.28, 0.33],

            [tick_value, tick_value],

            color="black",

            linewidth=0.45,

            zorder=5,

        )

        legend_ax.text(

            0.36,

            tick_value,

            str(tick_label),

            ha="left",

            va="center",

            fontsize=LEGEND_FONTSIZE,

            fontfamily=BASE_FONT,

            color="black",

            zorder=6,

            clip_on=False,

        )

    return legend_ax


def add_shared_elevation_legend(ax):

    add_boxed_gradient_legend(

        ax=ax,

        cmap=ELEV_CMAP,

        norm_for_bar=shared_elev_norm,

        tick_values=shared_elev_tick_values,

        tick_labels_for_bar=shared_elev_tick_labels,

        title="Elevation",

        width="36%",

        height="39%",

        loc="lower left",

        borderpad=0.78,

    )


def make_habitat_handles():

    handles = [

        mpatches.Patch(

            facecolor=NOT_ANALYZED_COLOR,

            edgecolor="#BDBDBD",

            label="Not analyzed",

        )

    ]

    for habitat, color in HABITAT_COLORS.items():

        handles.append(

            mpatches.Patch(

                facecolor=color,

                edgecolor="black",

                linewidth=0.25,

                label=HABITAT_DISPLAY[habitat],

            )

        )

    return handles


def make_cell_center_xy(transform, height, width):

    cols = np.arange(width)

    rows = np.arange(height)

    xs = transform.c + (cols + 0.5) * transform.a + 0.5 * transform.b

    ys = transform.f + 0.5 * transform.d + (rows + 0.5) * transform.e

    x_grid, y_grid = np.meshgrid(xs, ys)

    return x_grid, y_grid


def load_dem_for_contours(dem_path, study_boundary_wgs84):

    if not os.path.exists(dem_path):

        raise FileNotFoundError(f"DEM raster not found:\n{dem_path}")

    print("Loading and downsampling DEM for contour map...")

    with rasterio.open(dem_path) as src:

        if src.crs is None:

            raise ValueError(

                "DEM raster has no CRS. Please define its CRS before plotting.")

        study_for_raster = study_boundary_wgs84.to_crs(src.crs)

        minx, miny, maxx, maxy = study_for_raster.total_bounds

        window = from_bounds(

            minx,

            miny,

            maxx,

            maxy,

            transform=src.transform,

        )

        window = window.round_offsets().round_lengths()

        out_height = max(2, int(window.height / DEM_DOWNSAMPLE_FACTOR))

        out_width = max(2, int(window.width / DEM_DOWNSAMPLE_FACTOR))

        dem = src.read(

            1,

            window=window,

            out_shape=(out_height, out_width),

            resampling=Resampling.bilinear,

            masked=True,

        )

        window_transform = src.window_transform(window)

        out_transform = window_transform * Affine.scale(

            window.width / out_width,

            window.height / out_height,

        )

        dem = np.asarray(dem, dtype=float)

        if src.nodata is not None:

            dem = np.where(dem == src.nodata, np.nan, dem)

        if np.ma.isMaskedArray(dem):

            dem = dem.filled(np.nan)

        study_mask = geometry_mask(

            geometries=list(study_for_raster.geometry),

            out_shape=dem.shape,

            transform=out_transform,

            invert=True,

        )

        dem = np.where(study_mask, dem, np.nan)

        x_grid, y_grid = make_cell_center_xy(

            out_transform,

            dem.shape[0],

            dem.shape[1],

        )

        if src.crs.to_string() != "EPSG:4326":

            transformer = Transformer.from_crs(

                src.crs, "EPSG:4326", always_xy=True)

            lon_grid, lat_grid = transformer.transform(x_grid, y_grid)

        else:

            lon_grid = x_grid

            lat_grid = y_grid

    valid = np.isfinite(dem)

    if not valid.any():

        raise ValueError(

            "No valid DEM values were found inside the study area.")

    print(

        f"DEM contour data loaded: shape={dem.shape}, "

        f"min={np.nanmin(dem):.1f} m, max={np.nanmax(dem):.1f} m"

    )

    return lon_grid, lat_grid, dem


# ------------------------------------------------------------

# 4. Load and prepare vector data

# ------------------------------------------------------------


print("Loading county and study boundary layers...")


counties = to_wgs84(read_shp(COUNTY_SHP, "County"), "County")

bbox = to_wgs84(read_shp(BBOX_SHP, "Bounding box"), "Bounding box")

study_boundary = dissolve_to_one(bbox)


county_name_col = find_county_name_col(counties)

counties["county_name"] = counties[county_name_col].astype(str)


counties_clip = safe_clip(

    counties,

    study_boundary,

    "counties clipped to study area",

)


print("Loading habitat layers...")


habitat_layers = []


for habitat, path in VEG_SHPS.items():

    gdf = to_wgs84(read_shp(path, habitat), habitat)

    gdf = safe_clip(gdf, study_boundary, habitat)

    gdf["habitat"] = habitat

    gdf["habitat_display"] = HABITAT_DISPLAY[habitat]

    habitat_layers.append(gdf[["habitat", "habitat_display", "geometry"]])


habitats = pd.concat(habitat_layers, ignore_index=True)

habitats = gpd.GeoDataFrame(habitats, geometry="geometry", crs="EPSG:4326")

habitats = clean_gdf(habitats)


print("Building Not analyzed layer...")


try:

    study_land = dissolve_to_one(counties_clip)

    habitat_union = union_geometry(habitats)

    not_analyzed_geom = study_land.geometry.iloc[0].difference(habitat_union)

    not_analyzed = gpd.GeoDataFrame(

        {"class": ["Not analyzed"]},

        geometry=[not_analyzed_geom],

        crs="EPSG:4326",

    )

    not_analyzed = clean_gdf(not_analyzed)


except Exception as e:

    print(f"Could not build Not analyzed layer: {e}")

    not_analyzed = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


print("Loading elevation class polygon layer...")


elev_path = ELEV_SHP if os.path.exists(ELEV_SHP) else ELEV_SHP_FALLBACK

elev = to_wgs84(read_shp(elev_path, "Elevation"), "Elevation")

elev = safe_clip(elev, study_boundary, "elevation")


elev_col = find_elevation_col(elev)

print(f"Using elevation column: {elev_col}")


elev = build_elevation_labels(elev, elev_col)


# DEM contour panel removed from Figure 1, so no DEM raster is loaded.

# Shared elevation scale for the elevation-class panel.

shared_elev_min = 0

shared_elev_max = int(

    np.ceil(float(elev["elev_upper_m"].max()) / 250.0) * 250

)


shared_elev_norm = mcolors.Normalize(

    vmin=shared_elev_min,

    vmax=shared_elev_max,

)


shared_elev_tick_values, shared_elev_tick_labels = make_shared_elevation_ticks(

    shared_elev_min,

    shared_elev_max,

    n_ticks=4,

)


print("\nShared elevation legend ticks:")

for v, lab in zip(shared_elev_tick_values, shared_elev_tick_labels):

    print(f"  {v}: {lab}")


unique_orders = sorted(elev["elev_order"].dropna().unique())


elev["elev_color"] = elev["elev_mid_m"].apply(

    lambda v: ELEV_CMAP(shared_elev_norm(v)) if np.isfinite(v) else "#FFFFFF"

)


filled_levels = np.arange(

    shared_elev_min,

    shared_elev_max + DEM_FILLED_CONTOUR_INTERVAL_M,

    DEM_FILLED_CONTOUR_INTERVAL_M,

)


line_levels = np.arange(

    shared_elev_min,

    shared_elev_max + DEM_LINE_CONTOUR_INTERVAL_M,

    DEM_LINE_CONTOUR_INTERVAL_M,

)


label_levels = np.arange(

    shared_elev_min,

    shared_elev_max + DEM_LABEL_CONTOUR_INTERVAL_M,

    DEM_LABEL_CONTOUR_INTERVAL_M,

)


# ------------------------------------------------------------

# 5. Drawing functions

# ------------------------------------------------------------


def draw_study_area_inset(ax, panel_label="(a)"):

    counties.plot(

        ax=ax,

        facecolor="#F5F5F5",

        edgecolor="#8A8A8A",

        linewidth=0.60,

    )

    study_boundary.boundary.plot(

        ax=ax,

        edgecolor="red",

        linewidth=2.0,

    )

    minx, miny, maxx, maxy = counties.total_bounds

    ax.set_xlim(minx - 0.5, maxx + 0.5)

    ax.set_ylim(miny - 0.5, maxy + 0.5)

    format_lonlat_axes(ax, label_size=TICK_LABEL_FONTSIZE)

    if panel_label is not None:

        add_panel_label(ax, panel_label)

    sxmin, symin, sxmax, symax = study_boundary.total_bounds

    ax.text(

        sxmin + 0.1,

        symax + 0.25,

        "Study area",

        fontsize=MAP_LABEL_FONTSIZE,

        color="red",

        fontfamily=BASE_FONT,

        fontweight="bold",

        ha="left",

        va="bottom",

    )


def draw_habitat_map(ax, panel_label="(b)", add_legend=True, legend_outside=True):

    if not not_analyzed.empty:

        not_analyzed.plot(

            ax=ax,

            facecolor=NOT_ANALYZED_COLOR,

            edgecolor="#D0D0D0",

            linewidth=0.12,

        )

    for habitat, color in HABITAT_COLORS.items():

        subset = habitats[habitats["habitat"] == habitat]

        if not subset.empty:

            subset.plot(

                ax=ax,

                facecolor=color,

                edgecolor="none",

                linewidth=0,

            )

    counties_clip.boundary.plot(

        ax=ax,

        edgecolor="#333333",

        linewidth=0.50,

    )

    study_boundary.boundary.plot(

        ax=ax,

        edgecolor="black",

        linewidth=0.9,

        clip_on=False,

    )

    add_county_labels(ax)

    set_study_extent(ax, study_boundary)

    format_lonlat_axes(ax, label_size=TICK_LABEL_FONTSIZE)

    if panel_label is not None:

        add_panel_label(ax, panel_label)

    if add_legend:

        habitat_handles = make_habitat_handles()

        if legend_outside:

            leg = ax.legend(

                handles=habitat_handles,

                loc="center left",

                bbox_to_anchor=(1.01, 0.5),

                fontsize=LEGEND_FONTSIZE,

                frameon=True,

                framealpha=1.0,

                borderpad=0.45,

                labelspacing=0.32,

                handlelength=1.1,

                handletextpad=0.42,

                prop={"family": BASE_FONT, "size": LEGEND_FONTSIZE},

            )

        else:

            leg = ax.legend(

                handles=habitat_handles,

                loc="lower left",

                fontsize=LEGEND_FONTSIZE,

                frameon=True,

                framealpha=1.0,

                borderpad=0.35,

                labelspacing=0.25,

                handlelength=1.0,

                handletextpad=0.35,

                prop={"family": BASE_FONT, "size": LEGEND_FONTSIZE},

            )

        leg.get_frame().set_facecolor("white")

        leg.get_frame().set_alpha(1.0)

        leg.get_frame().set_linewidth(0.55)


def draw_elevation_pixel_map(ax, panel_label="(c)", add_colorbar=True):

    for order in unique_orders:

        subset = elev[elev["elev_order"] == order]

        if not subset.empty:

            subset.plot(

                ax=ax,

                facecolor=subset["elev_color"],

                edgecolor="none",

                linewidth=0,

            )

    counties_clip.boundary.plot(

        ax=ax,

        edgecolor="#333333",

        linewidth=0.50,

    )

    study_boundary.boundary.plot(

        ax=ax,

        edgecolor="black",

        linewidth=0.9,

        clip_on=False,

    )

    add_geographic_labels(ax)

    set_study_extent(ax, study_boundary)

    format_lonlat_axes(ax, label_size=TICK_LABEL_FONTSIZE)

    if panel_label is not None:

        add_panel_label(ax, panel_label)

    if add_colorbar:

        add_shared_elevation_legend(ax)


def draw_dem_contour_map(ax, panel_label="(d)", add_colorbar=True):

    ax.contourf(

        dem_lon_grid,

        dem_lat_grid,

        dem_array,

        levels=filled_levels,

        cmap=ELEV_CMAP,

        norm=shared_elev_norm,

        alpha=0.92,

        extend="both",

    )

    ax.contour(

        dem_lon_grid,

        dem_lat_grid,

        dem_array,

        levels=line_levels,

        colors=DEM_CONTOUR_LINE_COLOR,

        linewidths=0.55,

        alpha=0.80,

    )

    label_contours = ax.contour(

        dem_lon_grid,

        dem_lat_grid,

        dem_array,

        levels=label_levels,

        colors=DEM_CONTOUR_LINE_COLOR,

        linewidths=0.75,

        alpha=0.90,

    )

    ax.clabel(

        label_contours,

        inline=True,

        fontsize=MAP_LABEL_FONTSIZE,

        fmt=lambda x: f"{int(x)}",

        colors=DEM_CONTOUR_LABEL_COLOR,

    )

    counties_clip.boundary.plot(

        ax=ax,

        edgecolor="#333333",

        linewidth=0.50,

        alpha=0.8,

    )

    study_boundary.boundary.plot(

        ax=ax,

        edgecolor="black",

        linewidth=0.9,

        clip_on=False,

    )

    add_geographic_labels(ax)

    set_study_extent(ax, study_boundary)

    format_lonlat_axes(ax, label_size=TICK_LABEL_FONTSIZE)

    if panel_label is not None:

        add_panel_label(ax, panel_label)

    if add_colorbar:

        add_shared_elevation_legend(ax)

    ax.text(

        0.985,

        0.025,

        f"Contours every {DEM_LINE_CONTOUR_INTERVAL_M} m",

        transform=ax.transAxes,

        ha="right",

        va="bottom",

        fontsize=LEGEND_FONTSIZE,

        fontfamily=BASE_FONT,

        color=DEM_CONTOUR_LINE_COLOR,

        bbox=dict(

            facecolor="white",

            edgecolor="#BDBDBD",

            linewidth=0.35,

            alpha=0.92,

            boxstyle="round,pad=0.18",

        ),

        zorder=30,

    )


# ------------------------------------------------------------

# 6. Save separate maps

# ------------------------------------------------------------


print("Creating separate Figure 1 maps...")


fig, ax = plt.subplots(figsize=(5.8, 6.2))

draw_study_area_inset(ax, panel_label="(a)")

plt.savefig(OUT_INSET_PNG, dpi=600, bbox_inches="tight")

plt.savefig(OUT_INSET_PDF, bbox_inches="tight")

plt.close()

print(f"Saved: {OUT_INSET_PNG}")


fig, ax = plt.subplots(figsize=(8.8, 5.4))

draw_habitat_map(ax, panel_label="(b)", add_legend=True, legend_outside=True)

plt.savefig(OUT_HABITAT_PNG, dpi=600, bbox_inches="tight")

plt.savefig(OUT_HABITAT_PDF, bbox_inches="tight")

plt.close()

print(f"Saved: {OUT_HABITAT_PNG}")


fig, ax = plt.subplots(figsize=(8.8, 5.4))

draw_elevation_pixel_map(ax, panel_label="(c)", add_colorbar=True)

plt.savefig(OUT_ELEV_PIXEL_PNG, dpi=600, bbox_inches="tight")

plt.savefig(OUT_ELEV_PIXEL_PDF, bbox_inches="tight")

plt.close()

print(f"Saved: {OUT_ELEV_PIXEL_PNG}")


# ------------------------------------------------------------

# 7. Save gridded 3-panel figure
# ------------------------------------------------------------

print("Creating gridded Figure 1 map...")

fig = plt.figure(figsize=(14.8, 10.2))

gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.0, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.12,
    hspace=0.22,
)

ax_study = fig.add_subplot(gs[:, 0])
ax_habitat = fig.add_subplot(gs[0, 1])
ax_elevation = fig.add_subplot(gs[1, 1])

draw_study_area_inset(ax_study, panel_label="(a)")
ax_study.set_title("Study Area", fontsize=MAP_TITLE_FONTSIZE,
                   fontfamily=BASE_FONT, pad=6)

draw_habitat_map(
    ax_habitat,
    panel_label="(b)",
    add_legend=True,
    legend_outside=False,
)
ax_habitat.set_title("Habitat Types", fontsize=MAP_TITLE_FONTSIZE,
                     fontfamily=BASE_FONT, pad=6)

draw_elevation_pixel_map(
    ax_elevation,
    panel_label="(c)",
    add_colorbar=True,
)
ax_elevation.set_title("Elevation Classes",
                       fontsize=MAP_TITLE_FONTSIZE, fontfamily=BASE_FONT, pad=6)

fig.suptitle(
    "Study Area, Habitat Types, and Elevation Context",
    fontsize=MAIN_TITLE_FONTSIZE,
    fontfamily=BASE_FONT,
    y=0.985,
)

fig.subplots_adjust(left=0.045, right=0.985, bottom=0.055, top=0.955)

plt.savefig(OUT_GRIDDED_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUT_GRIDDED_PDF, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT_GRIDDED_PNG}")
print("Done.")
