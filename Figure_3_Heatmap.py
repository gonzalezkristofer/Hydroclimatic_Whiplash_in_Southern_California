"""Create ONLY the two 0-6-month habitat/elevation lag heatmap PNGs.

Uses Pillow instead of matplotlib to avoid the rendering stop seen in VS Code.
Reads the SAME input CSVs and applies the SAME best-lag selection rule.
Does not overwrite the original Figure_3 folder or modify any CSV.
"""
from __future__ import annotations

import csv
import math
import re
import sys
import traceback
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(r"D:\Graduate Research\Final_Paper_Analysis")
CSV_DIR = (BASE_DIR / "Correlation_Time_Lag" / "Original_MultiK" /
           "ALL_Ks" / "Combined" / "3-Month" / "CSV_Summaries")
OUTPUT_DIR = BASE_DIR / "Figure_3_Colors"
DIRECTION = "Combined"
WINDOW = "3-Month"
P_THRESHOLD = 0.05
R_THRESHOLD = 0.30
LAG_MIN, LAG_MAX = 0, 6

VARIABLES = ["NDVI", "SMRz", "TMean", "TMax", "TMin", "VPDMax", "VPDMin"]
VARIABLE_LABELS = ["NDVI", "SMrz", "TMean", "TMax", "TMin", "VPDMax", "VPDMin"]
HABITATS = ["Agriculture", "Conifer_Forest", "Conifer_Woodland", "Desert",
            "Hardwood", "Herbaceous", "Shrub", "Urban"]
BLUE = ["#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#deebf7"]
GRAY = "#d9d9d9"


def font(size: int, bold: bool = False):
    # Font names are resolved by Pillow locally; no font files are distributed.
    candidates = (["C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else
                  ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_best(csv_file: Path, group_kind: str):
    if not csv_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")
    by_key = {}
    groups = set()
    with csv_file.open("r", newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        columns = set(rows.fieldnames or [])
        if group_kind == "habitat":
            group_col = "habitat"
        else:
            possibilities = ["elevation_bin", "elev_bin", "ELEV_BIN", "Elev_Bin",
                             "elevation_class", "elev_class", "Elevation", "elevation",
                             "Elev", "elev", "gridcode", "GRIDCODE", "Gridcode", "DN",
                             "VALUE", "Value", "value"]
            group_col = next((c for c in possibilities if c in columns), None)
        if not group_col or group_col not in columns:
            raise ValueError(
                f"No {group_kind} class column; CSV columns are {sorted(columns)}")
        mandatory = {"variable", "lag_months", "spearman_r", "p_value"}
        if not mandatory.issubset(columns):
            raise ValueError(
                f"Missing input columns: {sorted(mandatory - columns)}")
        for row in rows:
            if ("direction" in columns and row["direction"].strip() != DIRECTION) or (
                    "window" in columns and row["window"].strip() != WINDOW):
                continue
            variable = row["variable"].strip()
            group = row[group_col].strip()
            if variable not in VARIABLES or not group:
                continue
            groups.add(group)
            try:
                lag = int(float(row["lag_months"]))
                r = float(row["spearman_r"])
                p = float(row["p_value"])
            except (ValueError, TypeError):
                continue
            if not (LAG_MIN <= lag <= LAG_MAX and math.isfinite(r)):
                continue
            if "significant_main" in columns and row["significant_main"].strip():
                significant = row["significant_main"].strip(
                ).lower() in ("true", "1", "yes", "y", "t")
            else:
                significant = p < P_THRESHOLD and abs(r) >= R_THRESHOLD
            if not significant:
                continue
            key = (group, variable)
            # Strongest absolute R; where tied, select earliest lag.
            prior = by_key.get(key)
            if prior is None or (-abs(r), lag) < (-abs(prior["r"]), prior["lag"]):
                by_key[key] = {"r": r, "lag": lag}
    if not groups:
        raise ValueError(
            "No groups found after direction/window filtering; verify CSV data.")
    if group_kind == "habitat":
        group_order = [g for g in HABITATS if g in groups] + \
            sorted(groups - set(HABITATS))
        labels = [g.replace("_", " ") for g in group_order]
    else:
        def sort_key(g):
            m = re.search(r"-?\d+\.?\d*", g)
            return float(m.group()) if m else math.inf
        group_order = sorted(groups, key=sort_key)
        numeric = [float(g)
                   for g in group_order if re.fullmatch(r"\d+(?:\.0+)?", g)]
        coded = len(numeric) == len(group_order) and max(numeric) <= 50
        labels = []
        for g in group_order:
            if coded:
                low = int(float(g)) * 250
                labels.append(f"{low}-{low + 250} m")
            else:
                labels.append(g if "m" in g.lower() else f"{g} m")
    return by_key, group_order, labels


def centered(draw, xy, text, text_font, fill):
    draw.text(xy, str(text), anchor="mm", font=text_font, fill=fill)


def create_png(kind: str):
    csv_name = ("Habitat" if kind == "habitat" else "Elevation") + \
        "_Lagged_Correlations_Combined_3-Month.csv"
    best, groups, labels = load_best(CSV_DIR / csv_name, kind)
    n = len(groups)
    scale = 2  # High-resolution text and shapes; keeps memory usage modest.
    cell_w, cell_h = 110 * scale, 56 * scale
    top = 164 * scale
    left = (225 if kind == "habitat" else 174) * scale
    bottom = 125 * scale
    right = 172 * scale
    plot_w, plot_h = len(VARIABLES) * cell_w, n * cell_h
    width, height = left + plot_w + right, top + plot_h + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(18 * scale)
    subtitle_font = font(13 * scale)
    tick_font = font(12 * scale)
    number_font = font(10 * scale)
    r_font = font(10 * scale)
    axis_font = font(14 * scale)

    heading = "Habitat" if kind == "habitat" else "Elevation"
    xcenter = left + plot_w // 2
    centered(draw, (xcenter, 30 * scale),
             f"{heading} best precipitation-leading lag", title_font, "#151515")
    centered(draw, (xcenter, 56 * scale),
             "Combined, 3-Month, 0-6 month lag window", subtitle_font, "#151515")
    centered(draw, (xcenter, 79 * scale),
             "Threshold: p < 0.05 and |R| >= 0.30", subtitle_font, "#151515")

    for i, (group, label) in enumerate(zip(groups, labels)):
        y0 = top + i * cell_h
        # Draw y labels right-aligned.
        draw.text((left - 17 * scale, y0 + cell_h // 2), label,
                  anchor="rm", font=tick_font, fill="#222222")
        for j, var in enumerate(VARIABLES):
            x0 = left + j * cell_w
            entry = best.get((group, var))
            lag = entry["lag"] if entry else None
            r = entry["r"] if entry else None
            fill = BLUE[lag] if entry else GRAY
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h),
                           fill=fill, outline="#f8fbff", width=2 * scale)
            xc, yc = x0 + cell_w // 2, y0 + cell_h // 2
            if entry is None:
                centered(draw, (xc, yc), "n.s.", tick_font, "#555555")
            else:
                light = lag <= 3
                lag_color = "white" if light else "#111111"
                if r >= 0:
                    r_color = "#B7F4C4" if light else "#166534"
                else:
                    r_color = "#FFB4A2" if light else "#991B1B"
                centered(draw, (xc, yc - 12 * scale),
                         str(lag), number_font, lag_color)
                centered(draw, (xc, yc + 12 * scale),
                         f"R={r:.2f}", r_font, r_color)

    # Variable labels below the heatmap, rotated to match the original figure.
    for j, label in enumerate(VARIABLE_LABELS):
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_img = Image.new(
            "RGBA", (tw + 22 * scale, th + 22 * scale), (0, 0, 0, 0))
        ImageDraw.Draw(text_img).text((11 * scale - bbox[0], 11 * scale - bbox[1]), label,
                                      font=tick_font, fill="#222222")
        rotated = text_img.rotate(
            45, expand=True, resample=Image.Resampling.BICUBIC)
        xc = left + j * cell_w + cell_w // 2
        yy = top + plot_h + 8 * scale
        image.paste(rotated, (xc - rotated.width // 2, yy), rotated)

    # The original blue lag scale, including the non-significant class.
    cx0 = left + plot_w + 45 * scale
    bar_w = 28 * scale
    ncolor = 8
    bar_h = plot_h / ncolor
    for k in range(ncolor):
        col = BLUE[6 - k] if k < 7 else GRAY
        yy0 = int(top + k * bar_h)
        yy1 = int(top + (k+1) * bar_h)
        draw.rectangle((cx0, yy0, cx0 + bar_w, yy1),
                       fill=col, outline="#777777", width=1)
        tick = str(6-k) if k < 7 else "n.s."
        draw.text((cx0 + bar_w + 8 * scale, (yy0 + yy1)//2), tick,
                  font=tick_font, fill="#333333", anchor="lm")

    centered(draw, (left + plot_w // 2, height - 30 * scale),
             "Response variable", axis_font, "#222222")
    # Vertical label via a rotated auxiliary layer.
    y_label = "Habitat class" if kind == "habitat" else "Elevation class"
    tmp = Image.new("RGBA", (plot_h, 45 * scale), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((plot_h // 2, 20 * scale), y_label, font=axis_font,
                             anchor="mm", fill="#222222")
    tmp = tmp.rotate(90, expand=True)
    image.paste(tmp, (14 * scale, top + plot_h//2 - tmp.height//2), tmp)

    label = "Selected precipitation-leading lag (months)"
    temp = Image.new("RGBA", (plot_h, 50 * scale), (255, 255, 255, 0))
    ImageDraw.Draw(temp).text((plot_h // 2, 22 * scale), label, font=axis_font,
                              anchor="mm", fill="#222222")
    temp = temp.rotate(90, expand=True)
    image.paste(temp, (width - 64 * scale, top +
                plot_h//2 - temp.height//2), temp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / \
        f"{heading}_BestLag_Heatmap_Combined_3-Month_Lag0to6.png"
    image.save(output, format="PNG", optimize=True)
    print(f"SAVED PNG: {output} ({output.stat().st_size:,} bytes)", flush=True)
    return output


def main():
    print("Generating only 2 colored lag heatmap PNGs (no matplotlib or CSV output).", flush=True)
    print("Output folder:", OUTPUT_DIR, flush=True)
    for kind in ("habitat", "elevation"):
        try:
            create_png(kind)
        except Exception:
            print(f"FAILED to generate {kind} heatmap:", flush=True)
            traceback.print_exc()
    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
