"""
###############################################################################
### PRECIPITATION WHIPLASH SENSITIVITY ANALYSIS: THREE DETECTION METHODS
### JANUARY 1982-DECEMBER 2022
###############################################################################
### PURPOSE
###
### This script applies three alternative whiplash-event identification methods
### to the exact same 3-month precipitation series, dates, and thresholds. Keeping every
### other setting identical makes it possible to measure how sensitive event
### counts and downstream ecological results are to the detection algorithm.
###
### INPUT SERIES
###   1. 3-Month Mean precipitation only
###
### THRESHOLD DEFINITIONS
###   1. Fixed SD: dry Z <= -0.8; wet Z >= +0.8
###   2. Empirical percentile: dry <= 20th percentile; wet >= 80th percentile
###
### THREE DETECTION METHODS
###
### METHOD 1 -- Chronological_First_Opposite
###   Move chronologically through the series. Beginning at a qualifying dry or
###   wet month, retain the first later month reaching the opposite threshold
###   within 24 months. The endpoint becomes the next possible start. Events do
###   not overlap internally, but consecutive events may share one endpoint.
###
### METHOD 2 -- Shortest_First
###   Generate all opposite-threshold candidate pairs within 24 months. Select
###   non-overlapping events by increasing duration: K3 first, then K6, K12,
###   and K24. Within each class, shorter lags are selected first. Shared
###   endpoints are allowed.
###
### METHOD 3 -- Original_MultiK
###   Reproduce the original code: run separate greedy searches using maximum
###   horizons K=3, 6, 12, and 24; combine the results; remove exact duplicate
###   start-end pairs; and reassign retained events to exclusive duration bins
###   based on actual lag. Because the four searches are independent, nested or
###   overlapping events can remain when their start/end pairs differ.
###
### FINAL DURATION CLASSES FOR ALL METHODS
###   K3:   1-3 months
###   K6:   4-6 months
###   K12:  7-12 months
###   K24: 13-24 months
###
### OUTPUT STRUCTURE
###   <OUT_ROOT>\Chronological_First_Opposite\<Threshold>\...
###   <OUT_ROOT>\Shortest_First\<Threshold>\...
###   <OUT_ROOT>\Original_MultiK\<Threshold>\...
###
### The script also creates comparison workbooks containing event totals,
### shared events, method-specific events, and overlap diagnostics.
###############################################################################
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# =============================================================================
# ### USER CONFIGURATION
# =============================================================================

# Explicit 3-month precipitation input file.
# Update only this filename if your local file uses a slightly different name.
PRECIP_3_MONTH_PATH = Path(
    r"D:\Graduate Research\Graduate Research\Data\PRISM_Precipitation_1981_2024\Stats\PRISM_Precipitation_1981_2024_ZScores_3-MonthMean.xlsx"
)

# First sheet is used when this is None.
SHEET_NAME: Optional[str] = None

# Common analysis period determined by GIMMS NDVI availability.
ANALYSIS_START = pd.Timestamp("1982-01-01")
ANALYSIS_END = pd.Timestamp("2022-12-31")

# Only the 3-month rolling mean is analyzed.
ROLLING_WINDOWS = [3]

# Two requested threshold analyses.
THRESHOLD_METHODS = {
    "SD_0.8": {
        "kind": "fixed_sd",
        "lower": -0.8,
        "upper": 0.8,
        "description": "Fixed Z-score thresholds: dry <= -0.8; wet >= +0.8",
    },
    "Percentile_20_80": {
        "kind": "empirical_percentile",
        "lower_percentile": 0.20,
        "upper_percentile": 0.80,
        "description": "Empirical thresholds: dry <= 20th percentile; wet >= 80th percentile",
    },
}

# Exclusive, continuous duration classes.
DURATION_CLASSES: List[Tuple[int, int, int, str]] = [
    (1, 3, 3, "1-3 months"),
    (4, 6, 6, "4-6 months"),
    (7, 12, 12, "7-12 months"),
    (13, 24, 24, "13-24 months"),
]
MAX_TRANSITION_MONTHS = 24

DETECTION_METHODS = {
    "Chronological_First_Opposite": (
        "One chronological search; first opposite threshold within 24 months"
    ),
    "Shortest_First": (
        "All candidates ranked by duration class and lag; non-overlapping shortest events retained first"
    ),
    "Original_MultiK": (
        "Original independent K=3,6,12,24 greedy searches followed by exact-pair deduplication"
    ),
}
ORIGINAL_K_LIST = [3, 6, 12, 24]

# Output folder.
OUT_ROOT = Path(
    r"D:\Graduate Research\Final_Paper_Analysis\Whiplash Analysis Precipitation 1982_2022"
)

USE_ASCII_ARROWS = True
RUN_PLOTS = True


# =============================================================================
# ### DISPLAY SETTINGS
# =============================================================================

WINDOW_LABELS = {
    3: "3-Month Mean",
}
WINDOW_ORDER = [WINDOW_LABELS[w] for w in ROLLING_WINDOWS]

EVENT_COLOR = {
    "Dry->Wet": "#d62728",
    "Wet->Dry": "#1f77b4",
}

LINE_WIDTH = 0.7
MARKER_SIZE = 65
THRESHOLD_COLOR = "dimgray"
THRESHOLD_LW = 1.0


# =============================================================================
# ### GENERAL HELPERS
# =============================================================================


def safe_name(value: object) -> str:
    """Return a Windows-safe file or folder name."""
    text = re.sub(r'[<>:"/\\|?*]', "_", str(value))
    return text.strip().strip(".")


def safe_sheet_name(value: object) -> str:
    """Return an Excel-safe worksheet name."""
    text = re.sub(r"[\[\]:*?/\\]", "_", str(value))
    return text[:31]


def load_table(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    """Read an Excel or CSV table and clean its column names."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=(0 if sheet is None else sheet))
        if isinstance(df, dict):
            df = df[next(iter(df))]
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Create Date, Year, and Month columns."""
    out = df.copy()
    lower_to_original = {str(c).strip().lower(): c for c in out.columns}

    if "date" in lower_to_original:
        out["Date"] = pd.to_datetime(
            out[lower_to_original["date"]], errors="coerce"
        )
    else:
        year_col = lower_to_original.get("year")
        month_col = lower_to_original.get("month")
        if year_col is None or month_col is None:
            raise ValueError(
                "Input requires either Date or both Year and Month columns.")

        out["Date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(out[year_col], errors="coerce"),
                "month": pd.to_numeric(out[month_col], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )

    out = out.dropna(subset=["Date"]).copy()
    out["Date"] = out["Date"].dt.to_period("M").dt.to_timestamp()
    out["Year"] = out["Date"].dt.year.astype(int)
    out["Month"] = out["Date"].dt.month.astype(int)
    return out


def infer_z_column(df: pd.DataFrame) -> str:
    """Find the precipitation standardized-value column.

    Accepted names include common Z-score variants. Mean/mean is retained as
    a fallback only if that is the standardized-value column in the 3-month file.
    """
    exact_names = {
        "z",
        "zscore",
        "z_score",
        "z-score",
        "z_scores",
        "z-scores",
        "zscore_mean",
        "mean",
    }

    for col in df.columns:
        if str(col).strip().lower() in exact_names:
            return col

    for col in df.columns:
        name = str(col).strip().lower()
        if "z" in name and "score" in name:
            return col

    for col in df.columns:
        if str(col).strip().lower().startswith("z"):
            return col

    raise ValueError("Could not identify a Z-score column.")


def infer_habitat(df: pd.DataFrame) -> pd.Series:
    """Return habitat labels, or 'All' when no Habitat column exists."""
    habitat_col = next(
        (c for c in df.columns if str(c).strip().lower() == "habitat"),
        None,
    )
    if habitat_col is None:
        return pd.Series("All", index=df.index, dtype="object")

    habitat = df[habitat_col].astype(str).str.strip()
    habitat = habitat.mask(habitat.eq(""), "All")
    return habitat


def season_from_month(month: pd.Series) -> pd.Series:
    """Use November-April as wet season and May-October as dry season."""
    return pd.Series(
        np.where(month.isin([11, 12, 1, 2, 3, 4]), "Wet", "Dry"),
        index=month.index,
        dtype="object",
    )


def water_year(year: int, month: int) -> int:
    return int(year + (month >= 10))


def month_difference(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return whole calendar-month difference between two monthly dates."""
    return int((end.year - start.year) * 12 + (end.month - start.month))


def classify_duration(lag_months: int) -> Tuple[Optional[int], Optional[str]]:
    """Assign one exclusive duration class to a 1-24 month transition."""
    for lower, upper, class_k, class_label in DURATION_CLASSES:
        if lower <= lag_months <= upper:
            return class_k, class_label
    return None, None


# =============================================================================
# ### LOCATE THE 3-MONTH PRECIPITATION FILE
# =============================================================================


def resolve_input_path(configured_path: Path) -> Optional[Path]:
    """Return the configured 3-month path or search its folder for a match."""
    if configured_path.exists():
        return configured_path

    folder = configured_path.parent
    if not folder.exists():
        return None

    patterns = [
        re.compile(
            r".*precip.*3[-_ ]?month[-_ ]?mean.*\.xlsx$", re.IGNORECASE),
        re.compile(
            r".*precip.*z[-_ ]?scores?.*3[-_ ]?month.*\.xlsx$", re.IGNORECASE),
    ]

    files = list(folder.glob("*.xlsx"))
    for pattern in patterns:
        matches = [f for f in files if pattern.match(f.name)]
        if matches:
            selected = sorted(matches, key=lambda q: (
                len(q.name), q.name.lower()))[0]
            print(f"[AUTO-FIND] 3-Month Mean: {selected}")
            return selected

    print("[FILE NOT FOUND] Could not resolve the 3-Month Mean precipitation file.")
    print(f"  Configured path: {configured_path}")
    print(f"  Excel files currently in: {folder}")
    for candidate in sorted(files, key=lambda q: q.name.lower()):
        print(f"    - {candidate.name}")
    return None


def build_precipitation_paths() -> Dict[int, Path]:
    """Locate only the 3-month rolling-mean precipitation file."""
    resolved = resolve_input_path(PRECIP_3_MONTH_PATH)
    return {3: resolved} if resolved is not None else {}


# =============================================================================
# ### PREPARE THE COMMON 1982-2022 PRECIPITATION SERIES
# =============================================================================


def prepare_precipitation_table(
    raw: pd.DataFrame,
    window_number: int,
) -> pd.DataFrame:
    """Create one clean monthly precipitation table for the common period."""
    df = parse_dates(raw)
    z_col = infer_z_column(df)

    df["Habitat"] = infer_habitat(df)
    df["Z"] = pd.to_numeric(df[z_col], errors="coerce")
    df["Variable"] = "Precipitation"
    df["WindowNumber"] = int(window_number)
    df["Window"] = WINDOW_LABELS.get(
        window_number, f"{window_number}-Month Mean")
    df["Season"] = season_from_month(df["Month"])

    df = df.loc[
        df["Date"].between(ANALYSIS_START, ANALYSIS_END, inclusive="both")
    ].copy()

    df = df.dropna(subset=["Z"]).copy()
    df = df[
        [
            "Variable",
            "Habitat",
            "WindowNumber",
            "Window",
            "Date",
            "Year",
            "Month",
            "Season",
            "Z",
        ]
    ]

    # Duplicate habitat-month records would make transition timing ambiguous.
    duplicate_mask = df.duplicated(
        subset=["Habitat", "Window", "Date"], keep=False
    )
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["Habitat", "Window", "Date"]].head()
        raise ValueError(
            "Duplicate Habitat/Window/Date rows were found. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    return df.sort_values(["Habitat", "Date"]).reset_index(drop=True)


# =============================================================================
# ### THRESHOLDS
# =============================================================================


def apply_threshold_method(
    full_series: pd.DataFrame,
    method_name: str,
    method_config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add dry/wet threshold columns and an anomaly sign.

    Sign = -1 for dry, +1 for wet, and 0 for values between thresholds.
    Percentile thresholds are computed separately by Habitat x Window.
    """
    pieces: List[pd.DataFrame] = []
    threshold_rows: List[dict] = []

    for (habitat, window), group in full_series.groupby(
        ["Habitat", "Window"], sort=False
    ):
        g = group.sort_values("Date").copy()

        if method_config["kind"] == "fixed_sd":
            dry_threshold = float(method_config["lower"])
            wet_threshold = float(method_config["upper"])
        elif method_config["kind"] == "empirical_percentile":
            dry_threshold = float(
                g["Z"].quantile(method_config["lower_percentile"])
            )
            wet_threshold = float(
                g["Z"].quantile(method_config["upper_percentile"])
            )
        else:
            raise ValueError(
                f"Unknown threshold method: {method_config['kind']}")

        if not np.isfinite(dry_threshold) or not np.isfinite(wet_threshold):
            print(
                f"[WARN] Skipping {habitat}, {window}, {method_name}: "
                "threshold could not be calculated."
            )
            continue

        if dry_threshold >= wet_threshold:
            print(
                f"[WARN] Skipping {habitat}, {window}, {method_name}: "
                f"dry threshold {dry_threshold:.4f} is not below wet threshold "
                f"{wet_threshold:.4f}."
            )
            continue

        g["ThresholdMethod"] = method_name
        g["DryThreshold"] = dry_threshold
        g["WetThreshold"] = wet_threshold
        g["Sign"] = np.select(
            [g["Z"] <= dry_threshold, g["Z"] >= wet_threshold],
            [-1, 1],
            default=0,
        ).astype(int)

        threshold_rows.append(
            {
                "ThresholdMethod": method_name,
                "Habitat": habitat,
                "Window": window,
                "N_Months": len(g),
                "FirstDate": g["Date"].min(),
                "LastDate": g["Date"].max(),
                "DryThreshold": dry_threshold,
                "WetThreshold": wet_threshold,
                "DryMonths": int((g["Sign"] == -1).sum()),
                "WetMonths": int((g["Sign"] == 1).sum()),
                "NeutralMonths": int((g["Sign"] == 0).sum()),
            }
        )
        pieces.append(g)

    classified = pd.concat(
        pieces, ignore_index=True) if pieces else pd.DataFrame()
    thresholds = pd.DataFrame(threshold_rows)
    return classified, thresholds


# =============================================================================
# ### WHIPLASH DETECTION: THREE ALTERNATIVE METHODS
# =============================================================================


def _event_record(g: pd.DataFrame, i: int, j: int, detection_method: str,
                  search_k: Optional[int] = None) -> dict:
    """Create one standardized event record from two opposite threshold rows."""
    start_date = pd.Timestamp(g.loc[i, "Date"])
    end_date = pd.Timestamp(g.loc[j, "Date"])
    lag = month_difference(start_date, end_date)
    duration_k, duration_label = classify_duration(lag)
    if duration_k is None:
        raise ValueError(f"Invalid whiplash lag: {lag}")

    start_sign = int(g.loc[i, "Sign"])
    direction = (
        "Dry->Wet" if USE_ASCII_ARROWS else "Dry→Wet"
    ) if start_sign == -1 else (
        "Wet->Dry" if USE_ASCII_ARROWS else "Wet→Dry"
    )

    start_year = int(g.loc[i, "Year"])
    start_month = int(g.loc[i, "Month"])
    end_year = int(g.loc[j, "Year"])
    end_month = int(g.loc[j, "Month"])

    return {
        "DetectionMethod": detection_method,
        "Variable": "Precipitation",
        "Habitat": g.loc[i, "Habitat"],
        "WindowNumber": int(g.loc[i, "WindowNumber"]),
        "Window": g.loc[i, "Window"],
        "ThresholdMethod": g.loc[i, "ThresholdMethod"],
        "DryThreshold": float(g.loc[i, "DryThreshold"]),
        "WetThreshold": float(g.loc[i, "WetThreshold"]),
        "SearchK": search_k if search_k is not None else np.nan,
        "DurationClassK": int(duration_k),
        "DurationClass": duration_label,
        "StartDate": start_date,
        "EndDate": end_date,
        "StartYear": start_year,
        "StartMonth": start_month,
        "EndYear": end_year,
        "EndMonth": end_month,
        "StartWY": water_year(start_year, start_month),
        "EndWY": water_year(end_year, end_month),
        "LagMonths": int(lag),
        "Direction": direction,
        "StartSeason": g.loc[i, "Season"],
        "EndSeason": g.loc[j, "Season"],
        "StartZ": float(g.loc[i, "Z"]),
        "EndZ": float(g.loc[j, "Z"]),
    }


def _intervals_conflict(start_a: pd.Timestamp, end_a: pd.Timestamp,
                        start_b: pd.Timestamp, end_b: pd.Timestamp) -> bool:
    """True when event interiors overlap; touching at one endpoint is allowed."""
    return max(start_a, start_b) < min(end_a, end_b)


def detect_chronological_first_opposite(group: pd.DataFrame) -> List[dict]:
    """One chronological 24-month search using the first opposite threshold."""
    g = group.sort_values("Date").reset_index(drop=True)
    rows: List[dict] = []
    i = 0
    n = len(g)

    while i < n:
        start_sign = int(g.loc[i, "Sign"])
        if start_sign not in (-1, 1):
            i += 1
            continue

        target_sign = -start_sign
        matched: Optional[int] = None
        for j in range(i + 1, n):
            lag = month_difference(g.loc[i, "Date"], g.loc[j, "Date"])
            if lag < 1:
                continue
            if lag > MAX_TRANSITION_MONTHS:
                break
            if int(g.loc[j, "Sign"]) == target_sign:
                matched = j
                break

        if matched is None:
            i += 1
            continue

        rows.append(_event_record(
            g, i, matched, "Chronological_First_Opposite"
        ))
        # Shared endpoint allowed: endpoint becomes the next possible start.
        i = matched

    return rows


def detect_shortest_first(group: pd.DataFrame) -> List[dict]:
    """Select non-overlapping opposite-threshold pairs from shortest to longest."""
    g = group.sort_values("Date").reset_index(drop=True)
    candidates: List[dict] = []
    extreme_indices = [
        idx for idx in range(len(g)) if int(g.loc[idx, "Sign"]) in (-1, 1)
    ]

    for position, i in enumerate(extreme_indices):
        target_sign = -int(g.loc[i, "Sign"])
        for j in extreme_indices[position + 1:]:
            lag = month_difference(g.loc[i, "Date"], g.loc[j, "Date"])
            if lag < 1:
                continue
            if lag > MAX_TRANSITION_MONTHS:
                break
            if int(g.loc[j, "Sign"]) == target_sign:
                candidates.append(_event_record(
                    g, i, j, "Shortest_First"
                ))

    candidates.sort(key=lambda row: (
        row["DurationClassK"], row["LagMonths"],
        row["StartDate"], row["EndDate"]
    ))

    accepted: List[dict] = []
    for candidate in candidates:
        conflict = any(
            _intervals_conflict(
                candidate["StartDate"], candidate["EndDate"],
                existing["StartDate"], existing["EndDate"]
            )
            for existing in accepted
        )
        if not conflict:
            accepted.append(candidate)

    return sorted(accepted, key=lambda row: (row["StartDate"], row["EndDate"]))


def _original_greedy_search(group: pd.DataFrame, search_k: int) -> List[dict]:
    """Reproduce one cumulative-K greedy search from the original code."""
    # The original detector received only qualifying threshold months.
    g = group.loc[group["Sign"].isin(
        [-1, 1])].sort_values("Date").reset_index(drop=True)
    rows: List[dict] = []
    i = 0
    n = len(g)

    while i < n:
        start_sign = int(g.loc[i, "Sign"])
        target_sign = -start_sign
        j = i + 1
        found = False

        while j < n:
            lag = month_difference(g.loc[i, "Date"], g.loc[j, "Date"])
            if lag < 0:
                j += 1
                continue
            if lag > search_k:
                break
            if int(g.loc[j, "Sign"]) == target_sign:
                rows.append(_event_record(
                    g, i, j, "Original_MultiK", search_k=search_k
                ))
                i = j
                found = True
                break
            j += 1

        if not found:
            i += 1

    return rows


def detect_original_multik(group: pd.DataFrame) -> List[dict]:
    """Run the original four independent K searches and exact-pair deduplication."""
    rows: List[dict] = []
    for search_k in ORIGINAL_K_LIST:
        rows.extend(_original_greedy_search(group, search_k))

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    # Original sorting keeps the smallest search K for exact duplicate pairs.
    frame = (
        frame.sort_values(["StartDate", "EndDate", "SearchK"])
        .drop_duplicates(
            subset=["Variable", "Habitat", "Window", "ThresholdMethod",
                    "StartDate", "EndDate"],
            keep="first",
        )
        .sort_values(["StartDate", "EndDate"])
        .reset_index(drop=True)
    )
    # DurationClassK was already assigned from actual lag in _event_record.
    return frame.to_dict("records")


def validate_nonoverlap(events: pd.DataFrame, detection_method: str) -> None:
    """Validate non-overlap for the two methods designed to produce unique sequences."""
    if events.empty or detection_method == "Original_MultiK":
        return

    problems = []
    for keys, group in events.groupby(
        ["ThresholdMethod", "Habitat", "Window"], sort=False
    ):
        g = group.sort_values(["StartDate", "EndDate"]).reset_index(drop=True)
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                if _intervals_conflict(
                    g.loc[a, "StartDate"], g.loc[a, "EndDate"],
                    g.loc[b, "StartDate"], g.loc[b, "EndDate"]
                ):
                    problems.append({
                        "DetectionMethod": detection_method,
                        "ThresholdMethod": keys[0],
                        "Habitat": keys[1],
                        "Window": keys[2],
                        "Event1Start": g.loc[a, "StartDate"],
                        "Event1End": g.loc[a, "EndDate"],
                        "Event2Start": g.loc[b, "StartDate"],
                        "Event2End": g.loc[b, "EndDate"],
                    })
    if problems:
        raise RuntimeError(
            f"{detection_method} produced interior-overlapping events:\n" +
            pd.DataFrame(problems).head(10).to_string(index=False)
        )


def detect_all_events(classified_series: pd.DataFrame,
                      detection_method: str) -> pd.DataFrame:
    """Run one selected detection method for every Habitat x Window series."""
    detector_map = {
        "Chronological_First_Opposite": detect_chronological_first_opposite,
        "Shortest_First": detect_shortest_first,
        "Original_MultiK": detect_original_multik,
    }
    if detection_method not in detector_map:
        raise ValueError(f"Unknown detection method: {detection_method}")

    rows: List[dict] = []
    detector = detector_map[detection_method]
    for _, group in classified_series.groupby(["Habitat", "Window"], sort=False):
        rows.extend(detector(group))

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    dedupe_columns = [
        "DetectionMethod", "Variable", "Habitat", "Window",
        "ThresholdMethod", "StartDate", "EndDate", "Direction"
    ]
    events = (
        events.sort_values(["Habitat", "WindowNumber", "StartDate", "EndDate"])
        .drop_duplicates(subset=dedupe_columns, keep="first")
        .reset_index(drop=True)
    )

    valid_lag = events["LagMonths"].between(1, MAX_TRANSITION_MONTHS)
    valid_flip = np.sign(events["StartZ"]) * np.sign(events["EndZ"]) == -1
    valid_class = events["DurationClassK"].isin([3, 6, 12, 24])
    events = events.loc[valid_lag & valid_flip &
                        valid_class].reset_index(drop=True)

    validate_nonoverlap(events, detection_method)
    return events


def overlap_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    """List all event pairs whose interiors overlap within the same series."""
    columns = [
        "DetectionMethod", "ThresholdMethod", "Habitat", "Window",
        "Event1Start", "Event1End", "Event1K", "Event1Direction",
        "Event2Start", "Event2End", "Event2K", "Event2Direction",
        "Nested", "SharedStart", "SharedEnd"
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for keys, group in events.groupby(
        ["DetectionMethod", "ThresholdMethod", "Habitat", "Window"], sort=False
    ):
        g = group.sort_values(["StartDate", "EndDate"]).reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.loc[i], g.loc[j]
                if _intervals_conflict(a["StartDate"], a["EndDate"],
                                       b["StartDate"], b["EndDate"]):
                    nested = (
                        (a["StartDate"] <= b["StartDate"]
                         and a["EndDate"] >= b["EndDate"])
                        or
                        (b["StartDate"] <= a["StartDate"]
                         and b["EndDate"] >= a["EndDate"])
                    )
                    rows.append({
                        "DetectionMethod": keys[0],
                        "ThresholdMethod": keys[1],
                        "Habitat": keys[2],
                        "Window": keys[3],
                        "Event1Start": a["StartDate"],
                        "Event1End": a["EndDate"],
                        "Event1K": a["DurationClassK"],
                        "Event1Direction": a["Direction"],
                        "Event2Start": b["StartDate"],
                        "Event2End": b["EndDate"],
                        "Event2K": b["DurationClassK"],
                        "Event2Direction": b["Direction"],
                        "Nested": nested,
                        "SharedStart": a["StartDate"] == b["StartDate"],
                        "SharedEnd": a["EndDate"] == b["EndDate"],
                    })
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# ### SUMMARIES
# =============================================================================


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize event counts and timing by habitat, window, and class."""
    if events.empty:
        return pd.DataFrame()

    group_cols = [
        "DetectionMethod",
        "ThresholdMethod",
        "Habitat",
        "WindowNumber",
        "Window",
        "DurationClassK",
        "DurationClass",
    ]

    summary = (
        events.groupby(group_cols, dropna=False)
        .agg(
            Whiplash_Total=("Direction", "size"),
            Whiplash_DryToWet=(
                "Direction",
                lambda s: int((s == "Dry->Wet").sum()),
            ),
            Whiplash_WetToDry=(
                "Direction",
                lambda s: int((s == "Wet->Dry").sum()),
            ),
            Median_Lag_Months=("LagMonths", "median"),
            Mean_Lag_Months=("LagMonths", "mean"),
            Minimum_Lag_Months=("LagMonths", "min"),
            Maximum_Lag_Months=("LagMonths", "max"),
        )
        .reset_index()
    )

    cross_season = (
        events.assign(
            CrossSeason=events["StartSeason"].ne(events["EndSeason"])
        )
        .groupby(group_cols, dropna=False)["CrossSeason"]
        .mean()
        .reset_index(name="Frac_CrossSeason")
    )

    summary = summary.merge(cross_season, on=group_cols, how="left")
    return summary.sort_values(
        ["WindowNumber", "Habitat", "DurationClassK"]
    ).reset_index(drop=True)


def totals_by_window_and_class(events: pd.DataFrame) -> pd.DataFrame:
    """Create totals with rolling windows as rows and exclusive classes as columns."""
    class_order = [3, 6, 12, 24]

    if events.empty:
        out = pd.DataFrame({"Window": WINDOW_ORDER})
        for k in class_order:
            out[f"K{k}"] = 0
        out["Total"] = 0
        return out

    pivot = events.pivot_table(
        index=["WindowNumber", "Window"],
        columns="DurationClassK",
        values="Direction",
        aggfunc="size",
        fill_value=0,
    )

    for k in class_order:
        if k not in pivot.columns:
            pivot[k] = 0

    pivot = pivot[class_order].reset_index().sort_values("WindowNumber")
    pivot = pivot.rename(columns={k: f"K{k}" for k in class_order})
    pivot["Total"] = pivot[[f"K{k}" for k in class_order]].sum(axis=1)
    pivot = pivot.drop(columns="WindowNumber")

    grand = {"Window": "Grand Total"}
    for column in [f"K{k}" for k in class_order] + ["Total"]:
        grand[column] = int(pivot[column].sum())
    return pd.concat([pivot, pd.DataFrame([grand])], ignore_index=True)


# =============================================================================
# ### PLOTS
# =============================================================================


def _intensity_alpha(z_value: float, threshold: float) -> float:
    """Map anomaly magnitude beyond its threshold to a visible bar opacity."""
    excess = max(0.0, abs(float(z_value)) - abs(float(threshold)))
    return float(min(1.0, 0.45 + 0.30 * excess))


def plot_detection_bars(
    classified_series: pd.DataFrame,
    events: pd.DataFrame,
    detection_method: str,
    threshold_method: str,
    out_dir: Path,
) -> None:
    """Plot monthly precipitation anomalies as red/blue bars with event links."""
    if classified_series.empty:
        return

    for (habitat, window), group in classified_series.groupby(
        ["Habitat", "Window"], sort=False
    ):
        g = group.sort_values("Date").copy()
        sub_events = (
            events.loc[events["Habitat"].eq(
                habitat) & events["Window"].eq(window)].copy()
            if not events.empty else pd.DataFrame()
        )

        fig, ax = plt.subplots(figsize=(15, 5), dpi=200)
        dry_threshold = float(g["DryThreshold"].iloc[0])
        wet_threshold = float(g["WetThreshold"].iloc[0])

        colors = []
        alphas = []
        for _, row in g.iterrows():
            if row["Z"] <= dry_threshold:
                colors.append("#d62728")
                alphas.append(_intensity_alpha(row["Z"], dry_threshold))
            elif row["Z"] >= wet_threshold:
                colors.append("#1f77b4")
                alphas.append(_intensity_alpha(row["Z"], wet_threshold))
            else:
                colors.append("#bdbdbd")
                alphas.append(0.35)

        bars = ax.bar(g["Date"], g["Z"], width=25,
                      color=colors, edgecolor="none")
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.axhline(dry_threshold, linestyle="--",
                   color="#8c1d18", linewidth=1.0)
        ax.axhline(wet_threshold, linestyle="--",
                   color="#174a7e", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)

        # Link each detected start and end. Shared endpoints are shown naturally.
        if not sub_events.empty:
            for _, ev in sub_events.iterrows():
                color = "#7f0000" if ev["Direction"] == "Dry->Wet" else "#08306b"
                ax.plot(
                    [ev["StartDate"], ev["EndDate"]],
                    [ev["StartZ"], ev["EndZ"]],
                    color=color, linewidth=1.2, alpha=0.8, zorder=4,
                )
                ax.scatter(
                    [ev["StartDate"], ev["EndDate"]],
                    [ev["StartZ"], ev["EndZ"]],
                    color=color, s=20, zorder=5,
                )

        legend_handles = [
            Line2D([0], [0], color="#d62728",
                   linewidth=8, label="Dry anomaly"),
            Line2D([0], [0], color="#1f77b4",
                   linewidth=8, label="Wet anomaly"),
            Line2D([0], [0], color="#bdbdbd",
                   linewidth=8, label="Neutral month"),
            Line2D([0], [0], color="#7f0000",
                   linewidth=1.5, label="Dry->Wet event"),
            Line2D([0], [0], color="#08306b",
                   linewidth=1.5, label="Wet->Dry event"),
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=5)
        ax.set_title(
            f"Precipitation whiplash detections | {habitat} | {window} | {detection_method} | {threshold_method}\n"
            "January 1982-December 2022; darker bars indicate stronger anomalies"
        )
        ax.set_ylabel("Precipitation Z-score")
        ax.set_xlabel("Date")
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        fig.savefig(
            out_dir /
            f"Whiplash_Detections_{safe_name(habitat)}_{safe_name(window)}_{safe_name(detection_method)}_{safe_name(threshold_method)}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# ### SAVE ONE THRESHOLD ANALYSIS
# =============================================================================


def save_method_outputs(
    detection_method: str,
    threshold_method: str,
    classified_series: pd.DataFrame,
    thresholds: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save outputs for one detection-method x threshold-method combination."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_name(detection_method)}_{safe_name(threshold_method)}"

    classified_series.to_csv(
        out_dir / f"Precipitation_Timeseries_{threshold_method}_1982_2022.csv",
        index=False, encoding="utf-8-sig"
    )
    thresholds.to_csv(
        out_dir / f"Thresholds_{threshold_method}_1982_2022.csv",
        index=False, encoding="utf-8-sig"
    )
    events_path = out_dir / f"Whiplash_Events_{stem}_1982_2022.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")

    for _, _, class_k, class_label in DURATION_CLASSES:
        subset = (
            events.loc[events["DurationClassK"].eq(class_k)]
            if not events.empty else pd.DataFrame(columns=events.columns)
        )
        subset.to_csv(
            out_dir /
            f"Whiplash_Events_{stem}_K{class_k}_{safe_name(class_label)}.csv",
            index=False, encoding="utf-8-sig"
        )

    summary.to_csv(
        out_dir / f"Whiplash_Summary_{stem}_1982_2022.csv",
        index=False, encoding="utf-8-sig"
    )
    overlaps = overlap_diagnostics(events)
    overlaps.to_csv(
        out_dir / f"Overlap_Diagnostics_{stem}.csv",
        index=False, encoding="utf-8-sig"
    )

    workbook = out_dir / f"Whiplash_Results_{stem}_1982_2022.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Item": ["Detection method", "Threshold method", "Detection definition"],
            "Value": [
                detection_method,
                threshold_method,
                DETECTION_METHODS[detection_method],
            ],
        }).to_excel(writer, sheet_name="ReadMe", index=False)
        thresholds.to_excel(writer, sheet_name="Thresholds", index=False)
        classified_series.to_excel(
            writer, sheet_name="Monthly_Series", index=False)
        events.to_excel(writer, sheet_name="All_Events", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        overlaps.to_excel(
            writer, sheet_name="Overlap_Diagnostics", index=False)

        for _, _, class_k, class_label in DURATION_CLASSES:
            subset = (
                events.loc[events["DurationClassK"].eq(class_k)]
                if not events.empty else pd.DataFrame(columns=events.columns)
            )
            subset.to_excel(
                writer,
                sheet_name=safe_sheet_name(f"K{class_k}_{class_label}"),
                index=False,
            )

        totals_by_window_and_class(events).to_excel(
            writer, sheet_name="Totals_All", index=False
        )
        totals_by_window_and_class(
            events.loc[events["Direction"].eq(
                "Dry->Wet")] if not events.empty else events
        ).to_excel(writer, sheet_name="Totals_Dry_to_Wet", index=False)
        totals_by_window_and_class(
            events.loc[events["Direction"].eq(
                "Wet->Dry")] if not events.empty else events
        ).to_excel(writer, sheet_name="Totals_Wet_to_Dry", index=False)

    if RUN_PLOTS:
        plot_detection_bars(
            classified_series, events, detection_method, threshold_method, out_dir
        )

    print(f"[SAVED] {detection_method} | {threshold_method}: {events_path}")
    print(f"[SAVED] {detection_method} | {threshold_method}: {workbook}")


# =============================================================================
# ### THREE-METHOD COMPARISON PRODUCTS
# =============================================================================


def _event_key_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Return event identity columns used for exact cross-method comparisons."""
    cols = [
        "ThresholdMethod", "Habitat", "Window", "StartDate", "EndDate", "Direction"
    ]
    if events.empty:
        return pd.DataFrame(columns=cols)
    return events[cols].drop_duplicates().copy()


def compare_detection_methods(
    results: Dict[Tuple[str, str], Dict[str, pd.DataFrame]],
    out_root: Path,
) -> None:
    """Create totals and exact event-overlap comparisons across all three methods."""
    comparison_dir = out_root / "_Detection_Method_Comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    workbook = comparison_dir / "Whiplash_Three_Detection_Method_Comparison_1982_2022.xlsx"

    total_rows = []
    all_events = []
    all_overlaps = []

    for (detection_method, threshold_method), item in results.items():
        events = item["events"].copy()
        if not events.empty:
            all_events.append(events)
        diag = overlap_diagnostics(events)
        if not diag.empty:
            all_overlaps.append(diag)

        for direction_label, subset in [
            ("All", events),
            ("Dry->Wet", events.loc[events["Direction"].eq("Dry->Wet")]
             if not events.empty else events),
            ("Wet->Dry", events.loc[events["Direction"].eq("Wet->Dry")]
             if not events.empty else events),
        ]:
            totals = totals_by_window_and_class(subset)
            totals.insert(0, "DirectionSubset", direction_label)
            totals.insert(0, "ThresholdMethod", threshold_method)
            totals.insert(0, "DetectionMethod", detection_method)
            total_rows.append(totals)

    totals_long = pd.concat(
        total_rows, ignore_index=True) if total_rows else pd.DataFrame()
    events_long = pd.concat(
        all_events, ignore_index=True) if all_events else pd.DataFrame()
    overlaps_long = pd.concat(
        all_overlaps, ignore_index=True) if all_overlaps else pd.DataFrame()

    pairwise_rows = []
    unique_rows = []
    methods = list(DETECTION_METHODS)
    for threshold_method in THRESHOLD_METHODS:
        method_events = {
            method: _event_key_frame(
                results[(method, threshold_method)]["events"])
            for method in methods
        }
        key_cols = ["ThresholdMethod", "Habitat",
                    "Window", "StartDate", "EndDate", "Direction"]

        for i, method_a in enumerate(methods):
            keys_a = method_events[method_a]
            set_a = set(
                map(tuple, keys_a[key_cols].itertuples(index=False, name=None)))
            for method_b in methods[i + 1:]:
                keys_b = method_events[method_b]
                set_b = set(
                    map(tuple, keys_b[key_cols].itertuples(index=False, name=None)))
                intersection = set_a & set_b
                union = set_a | set_b
                pairwise_rows.append({
                    "ThresholdMethod": threshold_method,
                    "MethodA": method_a,
                    "MethodB": method_b,
                    "EventsA": len(set_a),
                    "EventsB": len(set_b),
                    "ExactSharedEvents": len(intersection),
                    "OnlyA": len(set_a - set_b),
                    "OnlyB": len(set_b - set_a),
                    "JaccardSimilarity": (len(intersection) / len(union)) if union else np.nan,
                })

        all_sets = {
            method: set(
                map(tuple, method_events[method][key_cols].itertuples(index=False, name=None)))
            for method in methods
        }
        union_all = set().union(*all_sets.values()) if all_sets else set()
        for event_key in sorted(union_all, key=str):
            row = dict(zip(key_cols, event_key))
            for method in methods:
                row[method] = event_key in all_sets[method]
            row["NumberOfMethods"] = sum(bool(row[m]) for m in methods)
            unique_rows.append(row)

    pairwise = pd.DataFrame(pairwise_rows)
    event_membership = pd.DataFrame(unique_rows)

    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        readme = pd.DataFrame({
            "DetectionMethod": list(DETECTION_METHODS.keys()),
            "Definition": list(DETECTION_METHODS.values()),
        })
        readme.to_excel(writer, sheet_name="Detection_Methods", index=False)
        totals_long.to_excel(writer, sheet_name="Totals_Long", index=False)
        pairwise.to_excel(
            writer, sheet_name="Pairwise_Exact_Matches", index=False)
        event_membership.to_excel(
            writer, sheet_name="Event_Membership", index=False)
        overlaps_long.to_excel(
            writer, sheet_name="Overlap_Diagnostics", index=False)
        if not events_long.empty:
            events_long.to_excel(
                writer, sheet_name="All_Events_Long", index=False)

    totals_long.to_csv(
        comparison_dir / "Three_Method_Totals_Long.csv",
        index=False, encoding="utf-8-sig"
    )
    pairwise.to_csv(
        comparison_dir / "Three_Method_Pairwise_Exact_Matches.csv",
        index=False, encoding="utf-8-sig"
    )
    event_membership.to_csv(
        comparison_dir / "Three_Method_Event_Membership.csv",
        index=False, encoding="utf-8-sig"
    )
    overlaps_long.to_csv(
        comparison_dir / "Three_Method_Overlap_Diagnostics.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"[SAVED] Three-method comparison: {workbook}")


# =============================================================================
# ### MAIN WORKFLOW
# =============================================================================


def main() -> None:
    print("\n" + "=" * 79)
    print("PRECIPITATION WHIPLASH: THREE DETECTION METHODS, 3-MONTH MEAN, 1982-2022")
    print("=" * 79)
    print(f"3-month precipitation file: {PRECIP_3_MONTH_PATH}")

    precipitation_paths = build_precipitation_paths()
    if not precipitation_paths:
        raise FileNotFoundError(
            "The 3-month precipitation file was not found. "
            "Check PRECIP_3_MONTH_PATH."
        )

    print("\nLocated precipitation files:")
    for window, path in sorted(precipitation_paths.items()):
        print(f"  {WINDOW_LABELS.get(window, window)}: {path}")

    missing = [w for w in ROLLING_WINDOWS if w not in precipitation_paths]
    if missing:
        print("[WARN] Missing requested windows: " + ", ".join(
            WINDOW_LABELS.get(w, str(w)) for w in missing
        ))

    prepared_tables: List[pd.DataFrame] = []
    for window, path in sorted(precipitation_paths.items()):
        try:
            raw = load_table(path, SHEET_NAME)
            prepared = prepare_precipitation_table(raw, window)
        except Exception as exc:
            print(
                f"[WARN] Skipping {WINDOW_LABELS.get(window, window)}: {exc}")
            continue
        if prepared.empty:
            print(
                f"[WARN] No usable rows for {WINDOW_LABELS.get(window, window)}")
            continue
        prepared_tables.append(prepared)
        print(
            f"[OK] {WINDOW_LABELS.get(window, window)}: {len(prepared):,} rows; "
            f"{prepared['Date'].min():%Y-%m} to {prepared['Date'].max():%Y-%m}"
        )

    if not prepared_tables:
        raise RuntimeError(
            "No usable precipitation data remained after preparation.")

    full_series = pd.concat(prepared_tables, ignore_index=True).sort_values(
        ["WindowNumber", "Habitat", "Date"]
    ).reset_index(drop=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    full_series.to_csv(
        OUT_ROOT / "Precipitation_3MonthMean_1982_2022.csv",
        index=False, encoding="utf-8-sig"
    )

    # Threshold classification is calculated once and reused by all detectors.
    classified_by_threshold: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for threshold_method, threshold_config in THRESHOLD_METHODS.items():
        classified_by_threshold[threshold_method] = apply_threshold_method(
            full_series, threshold_method, threshold_config
        )

    results: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    for detection_method, detection_description in DETECTION_METHODS.items():
        print("\n" + "=" * 79)
        print(f"DETECTION METHOD: {detection_method}")
        print(detection_description)

        for threshold_method, threshold_config in THRESHOLD_METHODS.items():
            print("-" * 79)
            print(f"Threshold: {threshold_method}")
            print(threshold_config["description"])

            classified, thresholds = classified_by_threshold[threshold_method]
            events = detect_all_events(classified, detection_method)
            summary = summarize_events(events)

            output_dir = OUT_ROOT / detection_method / threshold_method
            save_method_outputs(
                detection_method, threshold_method,
                classified, thresholds, events, summary, output_dir
            )

            print(
                f"[RESULT] {detection_method} | {threshold_method}: "
                f"{len(events):,} total events"
            )
            if not events.empty:
                counts = events.groupby(
                    ["Window", "DurationClassK", "DurationClass"]
                ).size().reset_index(name="Events")
                print(counts.to_string(index=False))
                overlap_count = len(overlap_diagnostics(events))
                print(
                    f"[DIAGNOSTIC] Interior-overlapping event pairs: {overlap_count}")

            results[(detection_method, threshold_method)] = {
                "classified": classified,
                "thresholds": thresholds,
                "events": events,
                "summary": summary,
            }

    compare_detection_methods(results, OUT_ROOT)

    print("\n" + "=" * 79)
    print("FINISHED")
    print(f"Outputs: {OUT_ROOT}")
    print("=" * 79 + "\n")


if __name__ == "__main__":
    main()
