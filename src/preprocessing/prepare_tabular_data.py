#!/usr/bin/env python
# coding: utf-8

# # Tabular Model Data Prepare
# 
# This notebook prepares cached full-history tabular inputs for the tabular-model pipeline directly from `raw_extract`.
# 
# Design:
# - prediction unit is one target imaging visit
# - prediction anchor is the previous imaging visit for the same patient
# - outputs are saved as 6 separate tables: `target`, `demographic`, `imaging_history`, `staging_history`, `mastectomy_history`, and `biomarker_history`
# - time-series history tables store all lag slots up to the maximum patient history length observed for that feature category
# - unused lag slots stay missing/null for patients with shorter histories
# - imaging history stores scan vectors and gap months since the previous imaging visit
# - staging / mastectomy / biomarker recency features are measured as float months before the anchor scan date
# - scan vectors follow `SCAN_TYPE_NAMES = [CT, MRI, PET, ULTRASOUND, XRAY, MAMMO_TOMO, NUCLEAR_MED]`
# - outputs are written to `data/prepared/multitask_tabular` by default
# 

# ## Setup
# 
# The next cell resolves paths, defines the cohort/filter settings, and lists which raw sources become baseline vs additional features.
# 

# In[2]:


import json
import os
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore")


def resolve_raw_extract_dir():
    repo_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("RAW_EXTRACT_DIR")
    candidates = [
        Path(configured).expanduser().resolve() if configured else None,
        repo_root / "data" / "raw_extract",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError("Set RAW_EXTRACT_DIR to the restricted raw-extract directory.")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_EXTRACT_DIR = resolve_raw_extract_dir()
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.environ.get("MULTITASK_TABULAR_DATA_DIR", REPO_ROOT / "data" / "prepared" / "multitask_tabular")
).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATHS = {
    "patient": RAW_EXTRACT_DIR / "patient.csv",
    "imaging_timeseries": RAW_EXTRACT_DIR / "imaging.csv",
    "staging_timeseries": RAW_EXTRACT_DIR / "staging.csv",
    "mastectomy_timeseries": RAW_EXTRACT_DIR / "surgery.csv",
    "biomarker_timeseries": RAW_EXTRACT_DIR / "biomarker.csv",
    "trajectories_category": RAW_EXTRACT_DIR / "trajectories_category.csv",
}

baselin_feeature = ["demographic", "imaging_timeseries", "staging_timeseries"]
addition_feature = ["mastectomy_timeseries", "biomarker_timeseries"]
USE_CATEGORY_FILTER = True
OUTPUT_TAG = "full_max_history_multitask_tabular"

INTERVAL_BINS = [3, 6, 12, 24]
INTERVAL_LABELS = ["0_3m", "3_6m", "6_12m", "12_24m", "24m_plus"]
SCAN_TYPE_NAMES = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
BIOMARKER_NAMES = ["ERBB2", "ESR1", "PGR"]
MASTECTOMY_TYPES = [
    "Biopsy of mastectomy scar",
    "Extended radical mastectomy",
    "Extended simple mastectomy",
    "Insertion of breast implant on same day of mastectomy (ie, immediate)",
    "Insertion or replacement of breast implant on separate day from mastectomy",
    "Mastectomy for gynecomastia",
    "Mastectomy with preservation of skin and nipple with synchronous implant",
    "Mastectomy, modified radical, including axillary lymph nodes, with or without pectoralis minor muscle, but excluding pectoralis major muscle",
    "Mastectomy, partial (eg, lumpectomy, tylectomy, quadrantectomy, segmentectomy)",
    "Mastectomy, partial (eg, lumpectomy, tylectomy, quadrantectomy, segmentectomy); with axillary lymphadenectomy",
    "Mastectomy, radical, including pectoral muscles, axillary lymph nodes",
    "Mastectomy, simple, complete",
    "Mastectomy, subcutaneous",
    "Modified radical mastectomy",
    "Partial mastectomy",
    "Partial mastectomy with axillary lymphadenectomy",
    "Placement of radiotherapy afterloading expandable catheter (single or multichannel) into the breast for interstitial radioelement application following partial mastectomy, includes imaging guidance; concurrent with partial mastectomy (List separately in addition to code for primary procedure)",
    "Placement of radiotherapy afterloading expandable catheter (single or multichannel) into the breast for interstitial radioelement application following partial mastectomy, includes imaging guidance; on date separate from partial mastectomy",
    "Radical mastectomy",
    "Radical mastectomy including axillary lymph nodes",
    "Radical mastectomy of bilateral breasts",
    "Simple mastectomy",
    "Simple mastectomy of bilateral breasts",
    "Simple mastectomy with axillary lymph node sampling",
    "Skin sparing mastectomy",
    "Subcutaneous mastectomy",
    "Total mastectomy and axillary dissection",
    "Total mastectomy and excision of part of pectoral muscles and chest wall",
    "Vertical pattern skin reducing mastectomy and reconstruction of breast",
]

required = [RAW_PATHS["patient"], RAW_PATHS["imaging_timeseries"], RAW_PATHS["staging_timeseries"]]
for raw_path in required:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing required raw file: {raw_path}")

active_feature_names = baselin_feeature + [name for name in addition_feature if name not in baselin_feeature]

print("RAW_EXTRACT_DIR:", RAW_EXTRACT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("baselin_feeature:", baselin_feeature)
print("addition_feature:", addition_feature)
print("history_mode: full max per feature table")


# ## Helpers
# 
# The next cell defines cohort filters, source-table preparation, anchor-based lag flattening, and table-building helpers.
# 

# In[3]:


def parse_bool_series(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna(False).map(lambda x: str(x).strip().lower() in truthy if not isinstance(x, bool) else x)


def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def days_to_months(start_date, end_date):
    return float((end_date - start_date).days) / 30.4375


def interval_to_bin(month_gap):
    if month_gap < 3:
        return "0_3m"
    if month_gap < 6:
        return "3_6m"
    if month_gap < 12:
        return "6_12m"
    if month_gap < 24:
        return "12_24m"
    return "24m_plus"


def normalize_stage_main(stage_name):
    if pd.isna(stage_name):
        return "unknown"
    s = str(stage_name).strip().upper()
    exclusion_terms = {"UNKNOWN", "NOT APPLICABLE", "NOT REPORTED", "STAGE X"}
    if s in exclusion_terms:
        return "unknown"
    if s.startswith("STAGE"):
        s = re.sub(r"^STAGE\s*", "", s).strip()
    if s.startswith("4") or s.startswith("IV"):
        return "IV"
    if s.startswith("3") or s.startswith("III"):
        return "III"
    if s.startswith("2") or s.startswith("II"):
        return "II"
    if s.startswith("1") or s.startswith("I"):
        return "I"
    if s.startswith("0"):
        return "0"
    return "unknown"


def normalize_stage_binary(stage_name):
    main_stage = normalize_stage_main(stage_name)
    if main_stage == "unknown":
        return "unknown"
    if main_stage == "IV":
        return "IV"
    return "0-III"


def safe_col_name(text):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_").lower()


def combine_scan_types(values):
    ordered = [str(v).strip().upper() for v in values if pd.notna(v) and str(v).strip()]
    unique = []
    for value in ordered:
        if value not in unique:
            unique.append(value)
    return "+".join(unique)


def encode_scan_vector(scan_label):
    parts = {part.strip().upper() for part in str(scan_label).split("+") if part.strip()}
    return [1 if name in parts else 0 for name in SCAN_TYPE_NAMES]


def select_patient_ids():
    patient_df = pd.read_csv(RAW_PATHS["patient"], dtype={"chai_patient_id": "string"})
    selected_ids = set(patient_df["chai_patient_id"].dropna().astype(str).unique())

    if USE_CATEGORY_FILTER:
        categories = pd.read_csv(RAW_PATHS["trajectories_category"], dtype={"chai_patient_id": "string"})
        mask = (
            parse_bool_series(categories["is_n_sequences_gt_1"])
            & parse_bool_series(categories["has_no_gap_gt_threshold"])
            & (
                parse_bool_series(categories["is_right_censor_3_year"])
                | parse_bool_series(categories["is_right_censor_4_year"])
                | parse_bool_series(categories["is_right_censor_5_year"])
                | parse_bool_series(categories["is_right_censor_more_than_5_year"])
            )
        )
        selected_ids = selected_ids & set(categories.loc[mask, "chai_patient_id"].dropna().astype(str).unique())

    return patient_df, selected_ids


def prepare_demographic(patient_df, selected_ids):
    out = patient_df[patient_df["chai_patient_id"].astype(str).isin(selected_ids)].copy()
    out = out[["chai_patient_id", "gender", "race", "year_of_birth", "age_years"]].copy()
    out = out.rename(columns={"chai_patient_id": "patient_id", "age_years": "age"})
    out = out.drop_duplicates(subset=["patient_id"]).sort_values("patient_id").reset_index(drop=True)
    return out


def prepare_imaging(selected_ids):
    imaging = pd.read_csv(RAW_PATHS["imaging_timeseries"], dtype={"chai_patient_id": "string"})
    imaging = imaging[imaging["chai_patient_id"].astype(str).isin(selected_ids)].copy()
    imaging["report_date"] = pd.to_datetime(imaging["report_date"], errors="coerce")
    imaging["sequence"] = pd.to_numeric(imaging["sequence"], errors="coerce").astype("Int64")
    imaging = imaging.dropna(subset=["report_date", "sequence", "scan_type_type"]).copy()
    imaging["scan_type_type"] = imaging["scan_type_type"].astype(str).str.upper().str.strip()
    out = (
        imaging.groupby(["chai_patient_id", "report_date", "sequence"], as_index=False)["scan_type_type"]
        .agg(combine_scan_types)
        .rename(columns={"chai_patient_id": "patient_id"})
        .sort_values(["patient_id", "report_date", "sequence"])
        .reset_index(drop=True)
    )
    out["scan_type_vector"] = out["scan_type_type"].map(encode_scan_vector)
    return out


def prepare_staging(selected_ids):
    staging = pd.read_csv(RAW_PATHS["staging_timeseries"], dtype={"chai_patient_id": "string"})
    staging = staging[staging["chai_patient_id"].astype(str).isin(selected_ids)].copy()
    staging["stage_date"] = pd.to_datetime(staging["stage_date"], errors="coerce")
    staging = staging.dropna(subset=["stage_date"]).copy()
    staging["stage_main"] = staging["stage_group_name"].apply(normalize_stage_main)
    staging["stage_binary"] = staging["stage_group_name"].apply(normalize_stage_binary)
    out = staging.rename(columns={"chai_patient_id": "patient_id"})
    out = out[["patient_id", "stage_date", "stage_group_name", "stage_main", "stage_binary"]].drop_duplicates()
    out = out.sort_values(["patient_id", "stage_date"]).reset_index(drop=True)
    return out


def prepare_mastectomy(selected_ids):
    surgery = pd.read_csv(RAW_PATHS["mastectomy_timeseries"], dtype={"chai_patient_id": "string"})
    surgery = surgery[surgery["chai_patient_id"].astype(str).isin(selected_ids)].copy()
    surgery["surgery_date"] = pd.to_datetime(surgery["surgery_date"], errors="coerce")
    surgery["surgery_name"] = surgery["surgery_name"].astype(str).str.strip()
    surgery = surgery[surgery["surgery_name"].isin(MASTECTOMY_TYPES)].copy()
    surgery["surgery_code"] = pd.to_numeric(surgery["surgery_code"], errors="coerce").astype("Int64")
    out = surgery.rename(columns={"chai_patient_id": "patient_id"})
    out = out[["patient_id", "surgery_date", "surgery_code", "surgery_name"]].drop_duplicates()
    out = out.sort_values(["patient_id", "surgery_date"]).reset_index(drop=True)
    return out


def prepare_biomarker(selected_ids):
    biomarker = pd.read_csv(RAW_PATHS["biomarker_timeseries"], dtype={"chai_patient_id": "string"})
    biomarker = biomarker[biomarker["chai_patient_id"].astype(str).isin(selected_ids)].copy()
    biomarker["report_date"] = pd.to_datetime(biomarker["report_date"], errors="coerce")
    biomarker["biomarker_name_name"] = biomarker["biomarker_name_name"].astype(str).str.strip().str.upper()
    biomarker["test_result_name"] = biomarker["test_result_name"].astype(str).str.strip()
    biomarker = biomarker[
        biomarker["biomarker_name_name"].isin(BIOMARKER_NAMES)
        & biomarker["test_result_name"].isin(["Positive", "Negative"])
    ].copy()
    out = biomarker.rename(columns={"chai_patient_id": "patient_id"})
    out = out[["patient_id", "report_date", "biomarker_name_code", "biomarker_name_name", "test_result_name"]].drop_duplicates()
    out = out.sort_values(["patient_id", "report_date"]).reset_index(drop=True)
    return out


def get_feature_tables(selected_ids, patient_df):
    tables = {"demographic": prepare_demographic(patient_df, selected_ids)}
    if "imaging_timeseries" in active_feature_names:
        tables["imaging_timeseries"] = prepare_imaging(selected_ids)
    if "staging_timeseries" in active_feature_names:
        tables["staging_timeseries"] = prepare_staging(selected_ids)
    if "mastectomy_timeseries" in active_feature_names:
        tables["mastectomy_timeseries"] = prepare_mastectomy(selected_ids)
    if "biomarker_timeseries" in active_feature_names:
        tables["biomarker_timeseries"] = prepare_biomarker(selected_ids)
    return tables


KEY_COLS = ["patient_id", "anchor_scan_date", "anchor_sequence", "target_sequence"]


def max_rows_per_patient(df):
    if df.empty:
        return 0
    return int(df.groupby("patient_id").size().max())


def make_base_row(patient_id, anchor_row, target_row):
    anchor_date = anchor_row["report_date"]
    return {
        "patient_id": patient_id,
        "anchor_scan_date": anchor_date,
        "anchor_sequence": int(anchor_row["sequence"]),
        "target_sequence": int(target_row["sequence"]),
    }


def flatten_imaging_history(anchor_imaging, patient_id, anchor_row, target_row, max_lag):
    row = make_base_row(patient_id, anchor_row, target_row)
    row["imaging_history_count_before_anchor"] = int(len(anchor_imaging))
    anchor_imaging = anchor_imaging.sort_values(["report_date", "sequence"]).reset_index(drop=True)
    for lag in range(1, max_lag + 1):
        idx = len(anchor_imaging) - lag
        prefix = f"imaging_t_minus_{lag}"
        row[f"{prefix}_scan_vector"] = pd.NA
        row[f"{prefix}_gap_months_since_previous"] = pd.NA
        if idx >= 0:
            rec = anchor_imaging.iloc[idx]
            row[f"{prefix}_scan_vector"] = str(rec["scan_type_vector"])
            if idx == 0:
                row[f"{prefix}_gap_months_since_previous"] = 0.0
            else:
                prev_rec = anchor_imaging.iloc[idx - 1]
                row[f"{prefix}_gap_months_since_previous"] = days_to_months(prev_rec["report_date"], rec["report_date"])
    return row


def flatten_anchor_history(history_df, prefix, date_col, patient_id, anchor_row, target_row, value_cols, max_lag):
    row = make_base_row(patient_id, anchor_row, target_row)
    anchor_date = anchor_row["report_date"]
    history_df = history_df.sort_values(date_col).reset_index(drop=True)
    row[f"{prefix}_history_count_before_anchor"] = int(len(history_df))
    for lag in range(1, max_lag + 1):
        idx = len(history_df) - lag
        base = f"{prefix}_t_minus_{lag}"
        row[f"{base}_months_before_anchor"] = pd.NA
        for col in value_cols:
            row[f"{base}_{safe_col_name(col)}"] = pd.NA
        if idx >= 0:
            rec = history_df.iloc[idx]
            row[f"{base}_months_before_anchor"] = days_to_months(rec[date_col], anchor_date)
            for col in value_cols:
                row[f"{base}_{safe_col_name(col)}"] = rec[col]
    return row


def compute_max_history_by_table(feature_tables):
    empty = pd.DataFrame(columns=["patient_id"])
    return {
        "imaging_history": max_rows_per_patient(feature_tables.get("imaging_timeseries", empty)),
        "staging_history": max_rows_per_patient(feature_tables.get("staging_timeseries", empty)),
        "mastectomy_history": max_rows_per_patient(feature_tables.get("mastectomy_timeseries", empty)),
        "biomarker_history": max_rows_per_patient(feature_tables.get("biomarker_timeseries", empty)),
    }


def build_anchor_tables(feature_tables):
    demographic_df = feature_tables["demographic"].copy()
    imaging_df = feature_tables["imaging_timeseries"].copy()
    max_history_by_table = compute_max_history_by_table(feature_tables)

    target_rows = []
    imaging_history_rows = []
    staging_history_rows = []
    mastectomy_history_rows = []
    biomarker_history_rows = []

    demo_lookup = demographic_df.set_index("patient_id")

    for patient_id, imaging_group in imaging_df.groupby("patient_id", sort=True):
        imaging_group = imaging_group.sort_values(["report_date", "sequence"]).reset_index(drop=True)
        if len(imaging_group) < 2:
            continue

        demo = demo_lookup.loc[patient_id] if patient_id in demo_lookup.index else None
        staging_patient = feature_tables["staging_timeseries"][feature_tables["staging_timeseries"]["patient_id"] == patient_id].copy()
        mastectomy_patient = feature_tables.get("mastectomy_timeseries", pd.DataFrame(columns=["patient_id", "surgery_date", "surgery_code", "surgery_name"]))
        mastectomy_patient = mastectomy_patient[mastectomy_patient["patient_id"] == patient_id].copy()
        biomarker_patient = feature_tables.get("biomarker_timeseries", pd.DataFrame(columns=["patient_id", "report_date", "biomarker_name_code", "biomarker_name_name", "test_result_name"]))
        biomarker_patient = biomarker_patient[biomarker_patient["patient_id"] == patient_id].copy()

        for target_idx in range(1, len(imaging_group)):
            target_row = imaging_group.iloc[target_idx]
            anchor_row = imaging_group.iloc[target_idx - 1]
            anchor_date = anchor_row["report_date"]
            target_date = target_row["report_date"]
            anchor_imaging = imaging_group.iloc[:target_idx].copy()
            gap_months = days_to_months(anchor_date, target_date)

            target_payload = make_base_row(patient_id, anchor_row, target_row)
            target_payload.update(
                {
                    "target_scan_date": target_date,
                    "target_scan_type_type": target_row["scan_type_type"],
                    "target_scan_vector": str(target_row["scan_type_vector"]),
                    "target_gap_months": float(gap_months),
                    "target_interval_months": float(gap_months),
                    "target_interval_bin": interval_to_bin(gap_months),
                    "history_imaging_count_before_anchor": int(len(anchor_imaging)),
                }
            )
            if demo is not None:
                target_payload["age_at_anchor"] = float(anchor_date.year - float(demo.get("year_of_birth"))) if pd.notna(demo.get("year_of_birth")) else pd.NA
            else:
                target_payload["age_at_anchor"] = pd.NA
            target_rows.append(target_payload)

            imaging_history_rows.append(
                flatten_imaging_history(
                    anchor_imaging,
                    patient_id,
                    anchor_row,
                    target_row,
                    max_history_by_table["imaging_history"],
                )
            )

            prior_staging = staging_patient[staging_patient["stage_date"] <= anchor_date].copy()
            staging_history_rows.append(
                flatten_anchor_history(
                    prior_staging,
                    "staging",
                    "stage_date",
                    patient_id,
                    anchor_row,
                    target_row,
                    ["stage_group_name", "stage_main", "stage_binary"],
                    max_history_by_table["staging_history"],
                )
            )

            prior_mastectomy = mastectomy_patient[mastectomy_patient["surgery_date"] <= anchor_date].copy()
            mastectomy_history_rows.append(
                flatten_anchor_history(
                    prior_mastectomy,
                    "mastectomy",
                    "surgery_date",
                    patient_id,
                    anchor_row,
                    target_row,
                    ["surgery_code"],
                    max_history_by_table["mastectomy_history"],
                )
            )

            prior_biomarker = biomarker_patient[biomarker_patient["report_date"] <= anchor_date].copy()
            biomarker_history_rows.append(
                flatten_anchor_history(
                    prior_biomarker,
                    "biomarker",
                    "report_date",
                    patient_id,
                    anchor_row,
                    target_row,
                    ["biomarker_name_code", "test_result_name"],
                    max_history_by_table["biomarker_history"],
                )
            )

    target_df = pd.DataFrame(target_rows).sort_values(KEY_COLS).reset_index(drop=True)
    imaging_history_df = pd.DataFrame(imaging_history_rows).sort_values(KEY_COLS).reset_index(drop=True)
    staging_history_df = pd.DataFrame(staging_history_rows).sort_values(KEY_COLS).reset_index(drop=True)
    mastectomy_history_df = pd.DataFrame(mastectomy_history_rows).sort_values(KEY_COLS).reset_index(drop=True)
    biomarker_history_df = pd.DataFrame(biomarker_history_rows).sort_values(KEY_COLS).reset_index(drop=True)

    prepared_tables = {
        "target": target_df,
        "demographic": demographic_df.sort_values("patient_id").reset_index(drop=True),
        "imaging_history": imaging_history_df,
        "staging_history": staging_history_df,
        "mastectomy_history": mastectomy_history_df,
        "biomarker_history": biomarker_history_df,
    }
    return prepared_tables, max_history_by_table


# ## Build Prepared Tables
# 
# The next cell prepares the cohort from raw extracts and saves 6 anchor-based tables. The train/validation/test split is intentionally left to downstream modeling notebooks.
# 

# In[4]:


patient_df, selected_ids = select_patient_ids()
feature_tables = get_feature_tables(selected_ids, patient_df)
prepared_tables, max_history_by_table = build_anchor_tables(feature_tables)

def format_date_columns_for_export(df):
    out = df.copy()
    date_cols = [col for col in out.columns if col.endswith("_date")]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


artifact_rows = []
for table_name, df in prepared_tables.items():
    csv_path = OUTPUT_DIR / f"{table_name}.csv"
    stale_parquet_path = OUTPUT_DIR / f"{table_name}.parquet"
    export_df = format_date_columns_for_export(df)
    export_df.to_csv(csv_path, index=False)
    stale_parquet_path.unlink(missing_ok=True)
    prepared_tables[table_name] = export_df
    artifact_rows.append(
        {
            "artifact": csv_path.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "path": str(csv_path),
        }
    )

metadata = {
    "output_tag": OUTPUT_TAG,
    "baselin_feeature": baselin_feeature,
    "addition_feature": addition_feature,
    "history_mode": "full_max_history_per_feature_table",
    "max_history_by_table": max_history_by_table,
    "use_category_filter": USE_CATEGORY_FILTER,
    "table_design": "6 separate tables with anchor-based full max-history features; no train/val/test split is performed in this notebook",
    "anchor_definition": "for target scan k, the anchor is scan k-1; imaging history includes scans up to the anchor; non-imaging histories include records on or before the anchor date",
    "target_interval_definition": "target_interval_months and target_gap_months match the C+R sequence scripts: day difference between anchor and target scan divided by 30.4375",
    "imaging_gap_definition": "imaging_t_minus_j_gap_months_since_previous matches the C+R target interval scale: day difference from the previous known imaging visit divided by 30.4375; earliest known visit is 0.0",
    "non_imaging_recency_definition": "staging_t_minus_j_months_before_anchor, mastectomy_t_minus_j_months_before_anchor, and biomarker_t_minus_j_months_before_anchor are day differences from event date to anchor scan divided by 30.4375",
    "missing_history_definition": "patients with fewer than the table-specific maximum history length keep unused lag slots missing/null",
    "date_export_format": "date columns are exported as YYYY-MM-DD strings",
    "tabular_file_formats": ["csv"],
    "scan_vector_definition": {"scan_type_names": SCAN_TYPE_NAMES, "vector_example": "[1, 0, 0, 0, 0, 0, 0] means CT only"},
    "tables": {name: df.columns.tolist() for name, df in prepared_tables.items()},
    "counts": {
        "selected_patients": int(len(selected_ids)),
        "target_rows": int(len(prepared_tables["target"])),
        "demographic_rows": int(len(prepared_tables["demographic"])),
    },
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

summary_df = pd.DataFrame(artifact_rows + [{"artifact": "metadata.json", "rows": pd.NA, "columns": pd.NA, "path": str(OUTPUT_DIR / "metadata.json")}])

print("selected patients:", len(selected_ids))
print("target rows:", len(prepared_tables["target"]))
print("max_history_by_table:", max_history_by_table)
display(summary_df)
display(prepared_tables["target"].head())
display(prepared_tables["imaging_history"].head())


# In[5]:


# Quick schema check: each history table uses the table-specific max patient visit count.
expected_lag_schema_rows = []
for table_name, max_lag in max_history_by_table.items():
    if table_name == "imaging_history":
        features_per_lag = 2  # scan_vector + gap_months_since_previous
    elif table_name == "staging_history":
        features_per_lag = 4  # stage_group_name + stage_main + stage_binary + months_before_anchor
    elif table_name == "mastectomy_history":
        features_per_lag = 2  # surgery_code + months_before_anchor
    elif table_name == "biomarker_history":
        features_per_lag = 3  # biomarker_name_code + test_result_name + months_before_anchor
    expected_lag_schema_rows.append({
        "table": table_name,
        "max_lag_slots": int(max_lag),
        "features_per_lag": int(features_per_lag),
        "lag_feature_columns": int(max_lag * features_per_lag),
        "actual_columns": int(len(prepared_tables[table_name].columns)),
    })
schema_check_df = pd.DataFrame(expected_lag_schema_rows)
display(schema_check_df)


# ## Notes
# 
# How the anchor-based lagging works:
# - for target scan `k`, the prediction point is the `(k-1)`th imaging visit
# - `anchor_scan_date` is the `(k-1)`th scan date and `target_scan_date` is the `k`th scan date
# - imaging history includes the anchor scan itself, so `imaging_t_minus_1_*` always refers to the anchor visit
# - imaging history stores only `scan_vector`; the vector positions are `[CT, MRI, PET, ULTRASOUND, XRAY, MAMMO_TOMO, NUCLEAR_MED]`
# - imaging gap columns are backward-looking transition gaps between consecutive historical scans; they never use the future target date
# - staging / mastectomy / biomarker histories keep the all lag slots up to the table-specific maximum patient history length on or before the anchor scan date
# - if a patient has fewer than the table-specific maximum earlier records for a source, the unused lag slots stay missing
# - demographic data stays in its own patient-level table and is reused later when joined to target rows
# 
# To change what gets prepared:
# - edit `baselin_feeature`
# - edit `addition_feature`
# - adjust the max-history logic or cap the table-specific maximum lag
# - edit `OUTPUT_TAG`
# - rerun the notebook
# 
