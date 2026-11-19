# ============================================================
# STANDALONE FIGURE 2 REDRAW SCRIPT
#
# Maps all matching Figure 2 GeoTIFF folders with cleaner styling.
#
# Correlation maps:
# - blue-yellow-red ramp
# - faded county boundaries
# - darker gray land, white ocean/background
# - NO conifer contour
# - NO elevation contour
#
# Lag maps:
# - dark blue at 0 months, light blue at 6 months
# - faded county boundaries
# - darker gray land, white ocean/background
# - conifer forest/woodland contour only
# - NO elevation contour
#
# Makes four lag conifer contour versions:
# - yellow
# - green
# - orange
# - red
#
# Also makes combined manuscript-ready figure:
# - correlation maps on top
# - lag-time maps below
# - separate colorbars and conifer overlay legend
#
# Inputs searched under:
# D:\Graduate Research\Final_Paper_Analysis
#
# Outputs saved to:
# D:\Graduate Research\Final_Paper_Analysis\Figure 2
# ============================================================

from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re
import os


os.environ.setdefault(
    "GDAL_DATA",
    r"C:\Users\Owner\anaconda3\envs\myenv\Library\share\gdal"
)

os.environ.setdefault(
    "PROJ_LIB",
    r"C:\Users\Owner\anaconda3\envs\myenv\Library\share\proj"
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

GEOTIFF_SEARCH_ROOT = Path(r"D:\Graduate Research\Final_Paper_Analysis")
OUT_DIR = Path(r"D:\Graduate Research\Final_Paper_Analysis\Figure 2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOCAL_SHP = r"D:\Graduate Research\Graduate Research\Data\BoundingBox\BoundingBox.shp"

COUNTY_SHP = r"D:\Graduate Research\Graduate Research\Study Area Extent\ca_counties\CA_Counties.shp"

VEG_SHPS = {
    "Conifer_Forest": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Conifer_Forest_4km.shp",
    "Conifer_Woodland": r"D:\Graduate Research\Graduate Research\California Vegetation - WHR13 Types ArcGIS Pro\Updated_Vegetation\4km_Vegetation_Types\Conifer_Woodland_4km.shp",
}

PLOT_VARIABLE_ORDER = [
    "NDVI",
    "SMRz",
    "TMean",
    "TMax",
    "TMin",
    "VPDMax",
    "VPDMin",
]

MODE_KEY = "Main_AnySign_R030"

MAX_LAG = 6
LAGS = list(range(0, MAX_LAG + 1))

SAVE_PNG = True
SAVE_PDF = True

# Land/ocean styling.
OCEAN_BACKGROUND_COLOR = "white"

# Slightly darker gray land background.
LAND_FILL_COLOR = "#CFCFCF"

# Faded county boundaries, used on both correlation and lag maps.
ADD_COUNTY_BOUNDARIES = True
COUNTY_BOUNDARY_COLOR = "#4D4D4D"
COUNTY_LINEWIDTH = 0.35
COUNTY_ALPHA = 0.22

# No elevation contour line anywhere.
ADD_ELEVATION_CONTOUR = False

# Conifer contour versions for lag maps only.
# IMPORTANT:
# These are separate lag-map outputs, not stacked on the same map.
CONIFER_STYLE_OPTIONS = [
    {
        "name": "Conifer_Yellow",
        "label": "Conifer forest/woodland",
        "color": "#FFD92F",
        "linewidth": 1.25,
        "linestyle": "-",
        "alpha": 0.98,
    },
    {
        "name": "Conifer_Green",
        "label": "Conifer forest/woodland",
        "color": "#A6D854",
        "linewidth": 1.25,
        "linestyle": "-",
        "alpha": 0.98,
    },
    {
        "name": "Conifer_Orange",
        "label": "Conifer forest/woodland",
        "color": "#FC8D62",
        "linewidth": 1.25,
        "linestyle": "-",
        "alpha": 0.98,
    },
    {
        "name": "Conifer_Red",
        "label": "Conifer forest/woodland",
        "color": "#D7191C",
        "linewidth": 1.25,
        "linestyle": "-",
        "alpha": 0.98,
    },
]


# ============================================================
# 2. COLOR MAPS
# ============================================================

# Make no-data transparent so gray land and white ocean/background show through.
TRANSPARENT_BAD = (1.0, 1.0, 1.0, 0.0)

# Correlation color ramp:
# -1 = blue, 0 = yellow, +1 = red
CORR_CMAP = plt.get_cmap("RdYlBu_r").copy()
CORR_CMAP.set_bad(TRANSPARENT_BAD)

# Lag color ramp:
# 0 months = darkest blue
# 6 months = lightest blue
LAG_COLORS = [
    "#08306b",  # 0 month
    "#08519c",  # 1 month
    "#2171b5",  # 2 months
    "#4292c6",  # 3 months
    "#6baed6",  # 4 months
    "#9ecae1",  # 5 months
    "#deebf7",  # 6 months
]

LAG_CMAP = ListedColormap(LAG_COLORS)
LAG_CMAP.set_bad(TRANSPARENT_BAD)
LAG_NORM = BoundaryNorm(np.arange(-0.5, MAX_LAG + 1.5, 1), LAG_CMAP.N)


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def safe_name(text):
    text = str(text)
    text = text.replace("->", "To")
    text = text.replace("|", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_geometries(gdf):
    gdf = gdf.copy()
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def union_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def dissolve_to_one(gdf):
    if gdf is None or gdf.empty:
        return None

    return gpd.GeoDataFrame(
        geometry=[union_geometry(gdf)],
        crs=gdf.crs
    )


def safe_clip(gdf, mask_gdf, label):
    try:
        clipped = gpd.clip(gdf, mask_gdf)
        return clean_geometries(clipped)
    except Exception as e:
        print(f"Clip failed for {label}. Using unclipped layer. Error: {e}")
        return gdf


def lon_formatter(x, pos):
    if x < 0:
        return f"{abs(x):.1f}°W"
    return f"{x:.1f}°E"


def lat_formatter(y, pos):
    if y < 0:
        return f"{abs(y):.1f}°S"
    return f"{y:.1f}°N"


def format_map_axes(ax, crs, label_size=7):
    ax.set_facecolor(OCEAN_BACKGROUND_COLOR)

    is_geographic = False

    try:
        is_geographic = bool(crs.is_geographic)
    except Exception:
        is_geographic = False

    if is_geographic:
        ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

    ax.tick_params(labelsize=label_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")


# ============================================================
# 4. FIND ALL MATCHING GEOTIFF FOLDERS
# ============================================================

def folder_has_required_geotiffs(folder):
    folder = Path(folder)

    if not folder.exists():
        return False

    for var_name in PLOT_VARIABLE_ORDER:
        corr_file = folder / \
            f"{safe_name(var_name)}_{safe_name(MODE_KEY)}_corr.tif"
        lag_file = folder / \
            f"{safe_name(var_name)}_{safe_name(MODE_KEY)}_lag.tif"

        if not corr_file.exists() or not lag_file.exists():
            return False

    return True


def find_all_input_raster_dirs():
    target_name = f"{safe_name('NDVI')}_{safe_name(MODE_KEY)}_corr.tif"

    folders = []

    print("\nSearching for Figure 2 GeoTIFF folders under:")
    print(GEOTIFF_SEARCH_ROOT)

    for match in GEOTIFF_SEARCH_ROOT.rglob(target_name):
        folder = match.parent

        # Avoid using outputs inside the final Figure 2 folder if any copies exist there.
        if str(OUT_DIR).lower() in str(folder).lower():
            continue

        if folder_has_required_geotiffs(folder):
            folders.append(folder)

    folders = sorted(set(folders))

    if len(folders) == 0:
        raise FileNotFoundError(
            "Could not find any complete Figure 2 GeoTIFF folders.\n\n"
            f"Searched under:\n{GEOTIFF_SEARCH_ROOT}\n\n"
            "Expected files include:\n"
            f"  NDVI_{safe_name(MODE_KEY)}_corr.tif\n"
            f"  NDVI_{safe_name(MODE_KEY)}_lag.tif\n\n"
            "Make sure the original lagged-correlation script finished successfully "
            "and SAVE_RASTERS = True."
        )

    print("\nFound complete GeoTIFF folders:")
    for folder in folders:
        print(f"  - {folder}")

    return folders


def read_raster(input_raster_dir, var_name, map_kind):
    raster_name = f"{safe_name(var_name)}_{safe_name(MODE_KEY)}_{map_kind}.tif"
    raster_path = Path(input_raster_dir) / raster_name

    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster:\n{raster_path}")

    da = rxr.open_rasterio(raster_path, masked=True).squeeze()

    if "band" in da.dims:
        da = da.isel(band=0, drop=True)

    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da


def make_output_prefix(input_raster_dir):
    rel = Path(input_raster_dir).relative_to(GEOTIFF_SEARCH_ROOT)
    parts = list(rel.parts)

    if parts[-1].lower() == "geotiffs":
        parts = parts[:-1]

    prefix = "_".join(safe_name(p) for p in parts)
    return prefix


# ============================================================
# 5. BASE MAP AND CONIFER HELPERS
# ============================================================

def load_study_boundary():
    soc = gpd.read_file(SOCAL_SHP)

    if soc.crs is None:
        raise ValueError("Study boundary shapefile has no CRS.")

    soc = clean_geometries(soc)
    study_boundary = dissolve_to_one(soc)

    if study_boundary is None or study_boundary.empty:
        raise ValueError("Could not create study boundary.")

    return study_boundary


def load_counties_for_land_and_boundaries(study_boundary):
    if not os.path.exists(COUNTY_SHP):
        print(
            f"County shapefile not found. Land fill and county lines skipped: {COUNTY_SHP}")
        return None

    counties = gpd.read_file(COUNTY_SHP)

    if counties.crs is None:
        print("County shapefile has no CRS. Land fill and county lines skipped.")
        return None

    counties = clean_geometries(counties)
    counties = counties.to_crs(study_boundary.crs)
    counties = safe_clip(counties, study_boundary,
                         "county land fill/boundaries")

    if counties.empty:
        print("County layer is empty after clipping. Land fill and county lines skipped.")
        return None

    return counties


def load_conifer_outline(study_boundary):
    pieces = []

    for habitat_name, path in VEG_SHPS.items():
        if not os.path.exists(path):
            print(f"Missing conifer shapefile: {path}")
            continue

        gdf = gpd.read_file(path)

        if gdf.crs is None:
            print(f"Conifer shapefile has no CRS, skipping: {path}")
            continue

        gdf = clean_geometries(gdf)
        gdf = gdf.to_crs(study_boundary.crs)
        gdf = safe_clip(gdf, study_boundary, habitat_name)

        if not gdf.empty:
            pieces.append(gdf[["geometry"]])

    if len(pieces) == 0:
        print("No conifer polygons loaded. Skipping conifer outline.")
        return None

    conifer = gpd.GeoDataFrame(
        pd.concat(pieces, ignore_index=True),
        crs=study_boundary.crs
    )

    conifer = clean_geometries(conifer)
    conifer = dissolve_to_one(conifer)

    return conifer


def add_land_fill(ax, map_crs, counties):
    if counties is None:
        return

    try:
        counties_map = counties.to_crs(map_crs)

        # Land fill underneath raster.
        counties_map.plot(
            ax=ax,
            facecolor=LAND_FILL_COLOR,
            edgecolor="none",
            linewidth=0,
            zorder=1
        )

    except Exception as e:
        print(f"Could not add land fill: {e}")


def add_county_boundaries(ax, map_crs, counties):
    if counties is None or not ADD_COUNTY_BOUNDARIES:
        return

    try:
        counties_map = counties.to_crs(map_crs)

        # Faded county lines above raster.
        counties_map.boundary.plot(
            ax=ax,
            color=COUNTY_BOUNDARY_COLOR,
            linewidth=COUNTY_LINEWIDTH,
            alpha=COUNTY_ALPHA,
            zorder=5
        )

    except Exception as e:
        print(f"Could not add county boundaries: {e}")


def add_conifer_outline(ax, map_crs, conifer_outline, conifer_style):
    """
    Add one conifer forest/woodland contour color.
    This is only used on lag maps.
    """

    if conifer_outline is None or conifer_style is None:
        return

    try:
        conifer_outline.to_crs(map_crs).boundary.plot(
            ax=ax,
            color=conifer_style["color"],
            linewidth=conifer_style["linewidth"],
            linestyle=conifer_style["linestyle"],
            alpha=conifer_style["alpha"],
            zorder=8
        )

    except Exception as e:
        print(f"Could not add conifer outline: {e}")


def add_conifer_legend_to_empty_axis(ax, conifer_style):
    ax.set_axis_off()

    ax.text(
        0.00,
        0.95,
        "Map overlay",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

    y = 0.65

    ax.plot(
        [0.02, 0.26],
        [y, y],
        transform=ax.transAxes,
        color=conifer_style["color"],
        linewidth=1.8,
        linestyle=conifer_style["linestyle"],
        clip_on=False
    )

    ax.text(
        0.31,
        y,
        conifer_style["label"],
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9
    )


# ============================================================
# 6. SHARED PANEL PLOTTING HELPER
# ============================================================

def plot_one_panel(
    ax,
    da,
    var_name,
    map_kind,
    counties,
    conifer_outline=None,
    conifer_style=None
):
    ax.set_facecolor(OCEAN_BACKGROUND_COLOR)

    # Gray land under the raster.
    add_land_fill(
        ax=ax,
        map_crs=da.rio.crs,
        counties=counties
    )

    if map_kind == "corr":
        # Correlation maps get only the raster and faded county boundaries.
        # No conifer contour.
        # No elevation contour.
        im = da.plot(
            ax=ax,
            cmap=CORR_CMAP,
            vmin=-1,
            vmax=1,
            add_colorbar=False,
            zorder=3
        )

    elif map_kind == "lag":
        # Lag maps get the raster, faded county boundaries,
        # and conifer contour only.
        im = da.plot(
            ax=ax,
            cmap=LAG_CMAP,
            norm=LAG_NORM,
            add_colorbar=False,
            zorder=3
        )

    else:
        raise ValueError("map_kind must be 'corr' or 'lag'.")

    # Faded county lines on both correlation and lag maps.
    add_county_boundaries(
        ax=ax,
        map_crs=da.rio.crs,
        counties=counties
    )

    # Conifer contour ONLY on lag maps.
    if map_kind == "lag":
        add_conifer_outline(
            ax=ax,
            map_crs=da.rio.crs,
            conifer_outline=conifer_outline,
            conifer_style=conifer_style
        )

    ax.set_title(var_name, fontsize=11)
    format_map_axes(ax, da.rio.crs, label_size=7)

    return im


# ============================================================
# 7. FIGURE FUNCTIONS
# ============================================================

def save_gridded_figure(
    input_raster_dir,
    map_kind,
    out_png,
    out_pdf,
    panel_label,
    title,
    conifer_style=None
):
    if map_kind not in ["corr", "lag"]:
        raise ValueError("map_kind must be 'corr' or 'lag'.")

    print(f"\nCreating {map_kind} figure from:")
    print(input_raster_dir)

    if map_kind == "lag" and conifer_style is not None:
        print(f"Conifer style: {conifer_style['name']}")

    map_dict = {
        var_name: read_raster(input_raster_dir, var_name, map_kind)
        for var_name in PLOT_VARIABLE_ORDER
    }

    study_boundary = load_study_boundary()
    counties = load_counties_for_land_and_boundaries(study_boundary)

    # Only load conifer for lag maps.
    # This prevents contour lines from appearing on correlation maps.
    conifer_outline = None

    if map_kind == "lag" and conifer_style is not None:
        conifer_outline = load_conifer_outline(study_boundary)

    ncols = 3
    nrows = int(np.ceil(len(PLOT_VARIABLE_ORDER) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.1 * ncols, 4.2 * nrows)
    )

    axes = np.atleast_1d(axes).ravel()

    last_im = None

    for ax, var_name in zip(axes, PLOT_VARIABLE_ORDER):
        da = map_dict[var_name]

        last_im = plot_one_panel(
            ax=ax,
            da=da,
            var_name=var_name,
            map_kind=map_kind,
            counties=counties,
            conifer_outline=conifer_outline,
            conifer_style=conifer_style
        )

    first_empty_idx = len(PLOT_VARIABLE_ORDER)

    if first_empty_idx < len(axes):
        if map_kind == "lag" and conifer_style is not None:
            add_conifer_legend_to_empty_axis(
                axes[first_empty_idx],
                conifer_style
            )
        else:
            axes[first_empty_idx].set_axis_off()

    for ax in axes[first_empty_idx + 1:]:
        ax.set_axis_off()

    if last_im is not None:
        if map_kind == "corr":
            cbar = fig.colorbar(
                last_im,
                ax=axes.tolist(),
                orientation="horizontal",
                fraction=0.04,
                pad=0.05
            )

            cbar.set_label("Spearman R", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        else:
            cbar = fig.colorbar(
                last_im,
                ax=axes.tolist(),
                orientation="horizontal",
                fraction=0.04,
                pad=0.05,
                ticks=LAGS
            )

            cbar.set_label("PPT-leading response lag, months", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=15, y=0.985)

    fig.text(
        0.015,
        0.975,
        panel_label,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold"
    )

    if SAVE_PNG:
        fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved PNG: {out_png}")

    if SAVE_PDF:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved PDF: {out_pdf}")

    plt.close(fig)


def save_combined_corr_lag_figure(
    input_raster_dir,
    out_png,
    out_pdf,
    title,
    conifer_style
):
    """
    Saves one manuscript-ready combined figure:
    - correlation maps on top
    - lag-time maps below
    - one correlation colorbar
    - one lag colorbar
    - conifer overlay legend in the empty lag axis
    """

    print("\nCreating combined correlation + lag-time figure from:")
    print(input_raster_dir)

    print(f"Conifer style: {conifer_style['name']}")

    corr_dict = {
        var_name: read_raster(input_raster_dir, var_name, "corr")
        for var_name in PLOT_VARIABLE_ORDER
    }

    lag_dict = {
        var_name: read_raster(input_raster_dir, var_name, "lag")
        for var_name in PLOT_VARIABLE_ORDER
    }

    study_boundary = load_study_boundary()
    counties = load_counties_for_land_and_boundaries(study_boundary)
    conifer_outline = load_conifer_outline(study_boundary)

    ncols = 3
    section_rows = int(np.ceil(len(PLOT_VARIABLE_ORDER) / ncols))
    total_rows = section_rows * 2

    fig, axes = plt.subplots(
        total_rows,
        ncols,
        figsize=(5.1 * ncols, 4.0 * total_rows)
    )

    axes = np.atleast_2d(axes)

    corr_axes = axes[:section_rows, :].ravel()
    lag_axes = axes[section_rows:, :].ravel()

    last_corr_im = None
    last_lag_im = None

    for ax, var_name in zip(corr_axes, PLOT_VARIABLE_ORDER):
        da = corr_dict[var_name]

        last_corr_im = plot_one_panel(
            ax=ax,
            da=da,
            var_name=var_name,
            map_kind="corr",
            counties=counties,
            conifer_outline=None,
            conifer_style=None
        )

    first_empty_idx = len(PLOT_VARIABLE_ORDER)

    for ax in corr_axes[first_empty_idx:]:
        ax.set_axis_off()

    for ax, var_name in zip(lag_axes, PLOT_VARIABLE_ORDER):
        da = lag_dict[var_name]

        last_lag_im = plot_one_panel(
            ax=ax,
            da=da,
            var_name=var_name,
            map_kind="lag",
            counties=counties,
            conifer_outline=conifer_outline,
            conifer_style=conifer_style
        )

    if first_empty_idx < len(lag_axes):
        add_conifer_legend_to_empty_axis(
            lag_axes[first_empty_idx],
            conifer_style
        )

    for ax in lag_axes[first_empty_idx + 1:]:
        ax.set_axis_off()

    fig.suptitle(title, fontsize=16, y=0.992)

    fig.text(
        0.015,
        0.970,
        "(a) Correlation",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold"
    )

    fig.text(
        0.015,
        0.500,
        "(b) Lag Time",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold"
    )

    # Use subplots adjustment rather than tight_layout only.
    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        top=0.955,
        bottom=0.065,
        hspace=0.45,
        wspace=0.18
    )

    if last_corr_im is not None:
        cbar_corr = fig.colorbar(
            last_corr_im,
            ax=corr_axes.tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.035
        )

        cbar_corr.set_label("Spearman R", fontsize=10)
        cbar_corr.ax.tick_params(labelsize=8)

    if last_lag_im is not None:
        cbar_lag = fig.colorbar(
            last_lag_im,
            ax=lag_axes.tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.035,
            ticks=LAGS
        )

        cbar_lag.set_label("PPT-leading response lag, months", fontsize=10)
        cbar_lag.ax.tick_params(labelsize=8)

    if SAVE_PNG:
        fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved combined PNG: {out_png}")

    if SAVE_PDF:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"Saved combined PDF: {out_pdf}")

    plt.close(fig)


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":

    input_dirs = find_all_input_raster_dirs()

    for input_dir in input_dirs:
        prefix = make_output_prefix(input_dir)

        corr_title = (
            "Best Spearman correlation with PPT\n"
            "3-month, zscore, any sign, p < 0.05, abs(R) >= 0.30"
        )

        lag_title_base = (
            "Best PPT-leading response lag\n"
            "3-month, zscore, any sign, p < 0.05, abs(R) >= 0.30"
        )

        combined_title_base = (
            "Spatial Patterns of Precipitation-Response Correlation and Lag Time\n"
            "3-month, zscore, any sign, p < 0.05, abs(R) >= 0.30"
        )

        # ----------------------------------------------------
        # Correlation map:
        # NO conifer contour
        # NO elevation contour
        # faded county boundaries only
        # ----------------------------------------------------
        save_gridded_figure(
            input_raster_dir=input_dir,
            map_kind="corr",
            out_png=OUT_DIR /
            f"{prefix}_Figure2a_Correlation_RdYlBu_NoContour.png",
            out_pdf=OUT_DIR /
            f"{prefix}_Figure2a_Correlation_RdYlBu_NoContour.pdf",
            panel_label="(a)",
            title=corr_title,
            conifer_style=None
        )

        # ----------------------------------------------------
        # Lag maps:
        # conifer contour only, four color options
        # NO elevation contour
        # ----------------------------------------------------
        for conifer_style in CONIFER_STYLE_OPTIONS:
            style_name = safe_name(conifer_style["name"])

            lag_title = (
                lag_title_base
                + f"\nConifer contour: {conifer_style['name']}"
            )

            save_gridded_figure(
                input_raster_dir=input_dir,
                map_kind="lag",
                out_png=OUT_DIR /
                f"{prefix}_Figure2b_Lag_Blues_{style_name}.png",
                out_pdf=OUT_DIR /
                f"{prefix}_Figure2b_Lag_Blues_{style_name}.pdf",
                panel_label="(b)",
                title=lag_title,
                conifer_style=conifer_style
            )

            # ------------------------------------------------
            # Combined manuscript-ready figure:
            # correlation maps on top, lag maps below
            # makes one combined file for each conifer color
            # ------------------------------------------------
            combined_title = (
                combined_title_base
                + f"\nLag-map conifer contour: {conifer_style['name']}"
            )

            save_combined_corr_lag_figure(
                input_raster_dir=input_dir,
                out_png=OUT_DIR /
                f"{prefix}_Figure2_Combined_Correlation_Above_Lag_{style_name}.png",
                out_pdf=OUT_DIR /
                f"{prefix}_Figure2_Combined_Correlation_Above_Lag_{style_name}.pdf",
                title=combined_title,
                conifer_style=conifer_style
            )

    print("\nDone.")
