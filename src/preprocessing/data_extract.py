#!/usr/bin/env python
# coding: utf-8

# # Data Extraction
# 
# This notebook extracts features from the original Parquet files and saves them separately.  
# In a later step, these features will be combined into two datasets:
# - **time-dependent** features  
# - **time-invariant** features  
# 
# Both datasets will be exported to **CSV** and stored in **SQLite** tables for downstream model training.

# In[1]:


import os
import re
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from stage_utils import raw_tnm_to_stage

# The licensed Parquet files are not distributed with this repository. Set
# RAW_DATA_DIR to the directory containing patient.parquet, imaging.parquet,
# staging.parquet, and the other vendor tables.
REPO_ROOT = Path(__file__).resolve().parents[2]
raw_data_value = os.environ.get("RAW_DATA_DIR")
DATA_DIR = Path(raw_data_value).expanduser().resolve() if raw_data_value else REPO_ROOT / "data" / "raw"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Licensed Parquet directory not found: {DATA_DIR}. "
        "Place the authorized ConcertAI files in data/raw or set RAW_DATA_DIR."
    )

# Derived patient-level files remain restricted and are ignored by Git.
OUT_DIR = Path(os.environ.get("RAW_EXTRACT_DIR", REPO_ROOT / "data" / "raw_extract")).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = OUT_DIR.parent / "raw_data.sqlite"
conn = sqlite3.connect(SQLITE_PATH)

print("DATA_DIR  =", DATA_DIR)
print("OUT_DIR   =", OUT_DIR.resolve())
print("SQLITE_DB =", SQLITE_PATH.resolve())

def to_iso_date(s: pd.Series) -> pd.Series:
    """Convert datetime-like series to ISO date strings (YYYY-MM-DD), keep NaN for missing."""
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")

def write_csv_and_sqlite(df: pd.DataFrame, name: str):
    """
    Save one table to:
      1) ./data/raw_extract/{name}.csv
      2) SQLite table named {name} inside raw_data.sqlite
    """
    csv_path = OUT_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    df.to_sql(name, conn, if_exists="replace", index=False)

    if "chai_patient_id" in df.columns:
        with conn:
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{name}_chai_patient_id ON {name}(chai_patient_id);')

    print(f"[OK] {name}: rows={len(df):,}, cols={df.shape[1]}  ->  {csv_path}")


# ## Patient Static Table
# 
# - `ASSUME_END_DATE = 2025-12-31` is used **only** to compute age for filtering when `date_of_death` is missing.
# - When exporting to CSV / SQLite, `date_of_death` remains **missing** if it was missing in the source data (we do **not** impute it with `2025-12-31` in the output).
# - Keep only the selected columns, and filter patients to those with **age in [30, 100]**.
# - Create `cohort_ids` (the filtered patient IDs) and use it to filter **all downstream tables** consistently.

# In[2]:


p = DATA_DIR / "patient.parquet"
df = pd.read_parquet(p)

raw_patient_rows = len(df)

# Count and remove missing patient IDs
missing_patient_id_rows = df["chai_patient_id"].isna().sum()
df = df[df["chai_patient_id"].notna()].copy()

# Count and remove duplicate patient IDs
duplicate_patient_rows = df.duplicated(subset=["chai_patient_id"], keep="first").sum()
df = df.drop_duplicates(subset=["chai_patient_id"], keep="first").copy()

# Assume missing value of date_of_death as 2025-12-31 to calculate age
ASSUME_END_DATE = pd.Timestamp("2025-12-31")

# Parse dates
df["date_of_birth"] = pd.to_datetime(df.get("date_of_birth"), errors="coerce")
df["date_of_death"] = pd.to_datetime(df.get("date_of_death"), errors="coerce")

# For age calculation only: censor missing death at 2025-12-31
death_for_age = df["date_of_death"].fillna(ASSUME_END_DATE)

# Age in whole years (integer, floor)
age_days = (death_for_age - df["date_of_birth"]).dt.days
df["age_years"] = np.floor(age_days / 365.25).astype("Int64")

# Filter: DOB must exist, age in [30, 100]
missing_dob_rows = df["date_of_birth"].isna().sum()
df = df[df["date_of_birth"].notna()].copy()

missing_age_rows = df["age_years"].isna().sum()
df = df[df["age_years"].notna()].copy()

age_out_of_scope_rows = ((df["age_years"] < 30) | (df["age_years"] > 100)).sum()
df = df[(df["age_years"] >= 30) & (df["age_years"] <= 100)].copy()

# Keep only requested columns (no org_* fields, no is_deceased)
keep_cols = [
    "chai_patient_id",
    "gender", "race", "ethnicity",
    "year_of_birth",
    "date_of_birth",
    "date_of_death",   # keep original (can be missing)
    "age_years",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_patient = df[keep_cols].copy()

# Convert datetime to ISO date strings (YYYY-MM-DD); missing stays missing
df_patient["date_of_birth"] = to_iso_date(df_patient["date_of_birth"])
df_patient["date_of_death"] = to_iso_date(df_patient["date_of_death"])

write_csv_and_sqlite(df_patient, "patient")

# Cohort ids for filtering other tables
cohort_ids = set(df_patient["chai_patient_id"].dropna().astype(str).unique())

print(f"[INFO] raw_patient_rows={raw_patient_rows:,}")
print(f"[INFO] missing_patient_id_filtered={missing_patient_id_rows:,}")
print(f"[INFO] duplicate_patient_id_filtered={duplicate_patient_rows:,}")
print(f"[INFO] missing_dob_filtered={missing_dob_rows:,}")
print(f"[INFO] missing_age_filtered={missing_age_rows:,}")
print(f"[INFO] age_outside_30_100_filtered={age_out_of_scope_rows:,}")
print(f"[INFO] cohort_ids={len(cohort_ids):,}")


# ## Biomarker Time-series Table
# 
# - Filter rows using `cohort_ids` from the Patient Static Table.
# - Keep columns: `hai_patient_id`, `biomarker_name_name`, `test_result_name`, `report_date`.
# - Convert `report_date` to `YYYY-MM-DD` format.
# - Drop rows where `report_date` is missing (or cannot be parsed).

# In[3]:


p = DATA_DIR / "biomarker.parquet"
df = pd.read_parquet(p)

# Filter to the patient cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Keep required columns
keep_cols = ["chai_patient_id", "biomarker_name_code", "biomarker_name_name", "test_result_name", "report_date"]
keep_cols = [c for c in keep_cols if c in df.columns]
df_biomarker = df[keep_cols].copy()

# Convert date to YYYY-MM-DD
df_biomarker["report_date"] = to_iso_date(df_biomarker["report_date"])

# Drop rows without date (not usable as time-series)
df_biomarker = df_biomarker[df_biomarker["report_date"].notna()].copy()

write_csv_and_sqlite(df_biomarker, "biomarker")


# ## Care Goal Time-series Table
# 
# - Filter rows using `cohort_ids`.
# - Keep columns: `chai_patient_id`, `care_plan_intent_name`, `treatment_plan_start_date`, `treatment_plan_end_date`.
# - Convert `treatment_plan_start_date` and `treatment_plan_end_date` to `YYYY-MM-DD` format.

# In[4]:


p = DATA_DIR / "care_goal.parquet"
df = pd.read_parquet(p)

# Filter to our patient cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Keep required columns
keep_cols = [
    "chai_patient_id",
    "care_plan_intent_name",
    "treatment_plan_start_date",
    "treatment_plan_end_date",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_care_goal = df[keep_cols].copy()

# Convert dates to YYYY-MM-DD
df_care_goal["treatment_plan_start_date"] = to_iso_date(df_care_goal["treatment_plan_start_date"])
df_care_goal["treatment_plan_end_date"]   = to_iso_date(df_care_goal["treatment_plan_end_date"])

write_csv_and_sqlite(df_care_goal, "care_goal")


# ## Condition Time-series Table
# 
# - Filter rows using `cohort_ids`.
# - Keep columns: `chai_patient_id`, `diagnosis_date`, `diagnosis_code_code`, `diagnosis_code_name`.
# - Convert `diagnosis_date` to `YYYY-MM-DD` format.
# - Drop rows where `diagnosis_date` is missing (or cannot be parsed).

# In[5]:


p = DATA_DIR / "condition.parquet"
df = pd.read_parquet(p)

# Filter to our patient cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Keep required columns
keep_cols = [
    "chai_patient_id",
    "diagnosis_date",
    "diagnosis_code_code",
    "diagnosis_code_name",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_condition = df[keep_cols].copy()

# Convert date to YYYY-MM-DD
df_condition["diagnosis_date"] = to_iso_date(df_condition["diagnosis_date"])

# Drop rows without diagnosis date (not usable as time-series)
df_condition = df_condition[df_condition["diagnosis_date"].notna()].copy()

write_csv_and_sqlite(df_condition, "condition")


# ## Imaging Time-series Table
# 
# - Read `imaging.parquet` and filter rows using `cohort_ids`.
# - Keep columns: `chai_patient_id`, `report_date`, `imaging_type_code`, `imaging_type_name`.
# - Add a new column `scan_type_type` by mapping `imaging_type_name` into 9 categories:
#   `CT`, `MRI`, `PET`, `ULTRASOUND`, `XRAY`, `MAMMO_TOMO`, `NUCLEAR_MED`, `FLUORO_ANGIO_IR`, `DXA`.
# - Convert `report_date` to `YYYY-MM-DD` format.
# - Add `sequence`, a 1-based imaging order within each patient sorted by `report_date`.

# In[6]:


p = DATA_DIR / "imaging.parquet"
df = pd.read_parquet(p)

# Filter to cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Get distinct values
vals = (
    df["imaging_type_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

vals = sorted(vals)

print(f"Distinct imaging_type_name count = {len(vals)}\n")
for v in vals:
    print(v)

# Optional: save to a text file for easier sharing
# out_txt = OUT_DIR / "imaging_type_name_distinct.txt"
# out_txt.write_text("\n".join(vals), encoding="utf-8")
# print(f"\n[OK] Saved to: {out_txt}")


# In[7]:


p = DATA_DIR / "imaging.parquet"
df = pd.read_parquet(p)

raw_imaging_rows = len(df)
raw_imaging_patients = df["chai_patient_id"].dropna().astype(str).nunique()

# Filter to our patient cohort
before_patient_filter_rows = len(df)
before_patient_filter_patients = df["chai_patient_id"].dropna().astype(str).nunique()

df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

after_patient_filter_rows = len(df)
after_patient_filter_patients = df["chai_patient_id"].dropna().astype(str).nunique()

patient_filter_removed_rows = before_patient_filter_rows - after_patient_filter_rows
patient_filter_removed_patients = before_patient_filter_patients - after_patient_filter_patients

print(f"[INFO] raw_imaging_rows={raw_imaging_rows:,}")
print(f"[INFO] raw_imaging_patients={raw_imaging_patients:,}")
print(f"[INFO] imaging_rows_removed_by_patient_cohort_filter={patient_filter_removed_rows:,}")
print(f"[INFO] imaging_patients_removed_by_patient_cohort_filter={patient_filter_removed_patients:,}")
print(f"[INFO] imaging_rows_after_patient_cohort_filter={after_patient_filter_rows:,}")
print(f"[INFO] imaging_patients_after_patient_cohort_filter={after_patient_filter_patients:,}")

def classify_scan_type(name) -> object:
    """
    Map imaging_type_name into exactly 9 categories:
      CT, MRI, PET, ULTRASOUND, XRAY, MAMMO_TOMO, NUCLEAR_MED, FLUORO_ANGIO_IR, DXA
    If not matched -> return np.nan
    """
    if pd.isna(name):
        return np.nan
    s = str(name).upper().strip()

    # --- Explicit NON-IMAGING / NOT-A-SCAN patterns (keep NaN on purpose) ---
    if re.search(
        r"ANESTHESIA|SEDATION|LABORATORY|FLOW CYTOMETRY|TELEPHONE FOLLOW-UP|RESULT COMMENTS|"
        r"\bREPORT\b|IMAGING REPORT|RADIOLOGY REPORT|CONTRAST MEDIA|CONSULTATION|"
        r"COLONOSCOPY|ENDOSCOPY|ESOPHAGOGASTRODUODENOSCOPY|BRONCHOSCOPY|CYSTOSCOPY|"
        r"LARYNGOSCOPY|NASOPHARYNGOSCOPY|CARDIOVASCULAR FUNCTION TEST|ELECTROCARDIOGRAPH",
        s
    ):
        return np.nan

    # 1) Mammography / Tomosynthesis (before XRAY)
    if re.search(r"MAMMOG|TOMOSYNTH|TOMOSYNTHESIS|DUCTOGRAPH|SCINTIMAMMOGRAPH", s):
        return "MAMMO_TOMO"

    # # 2) DXA / Bone density
    # if re.search(r"\bDXA\b|\bDEXA\b|BONE DENSITY|ABSORPTIOMETRY", s):
    #     return "DXA"

    # 3) PET (before CT because PET-CT contains CT)
    if re.search(r"\bPET\b|POSITRON EMISSION|FDG|PSMA", s):
        return "PET"

    # 4) MRI / MR
    if re.search(r"\bMRI\b|MAGNETIC RESONANCE|\bMR ANGIO|\bMR VENOGRAPH|CHOLANGIOPANCREATOGRAPHY", s):
        return "MRI"

    # 5) CT / CAT / Computed Tomography (+ radiotherapy planning + x-ray tomography)
    if re.search(r"\bCT\b|COMPUTED TOMOGRAPH|CAT SCAN|LDCT|CT ANGIO|CT VENOGRAPH|RADIOTHERAPY PLANNING", s):
        return "CT"
    if re.search(r"X-?RAY TOMOGRAPH", s):
        return "CT"

    # 6) Ultrasound / Doppler / Duplex / Echo / Echography
    if re.search(r"ULTRASOUND|ULTRASONOGRAPH|DOPPLER|DUPLEX|ECHOCARDIO|ECHOGRAPH", s):
        # return "ULTRASOUND"
        return "ULTRASOUND" if re.search(r"\bBREAST\b", s) else np.nan

    # # 7) Fluoroscopy / Angio / Interventional + classic contrast series
    # if re.search(
    #     r"FLUORO|ANGIOGRAPH|VENOGRAPH|AORTOGRAPH|MYELOGRAM|ARTHROGRAPH|CHOLANGIOGRAPH|"
    #     r"FISTULOGRAPH|HYSTEROSALPING|INTERVENTIONAL RADIOLOGY|"
    #     r"BARIUM|SWALLOW|SMALL BOWEL|UPPER GASTROINTESTINAL|PYELOGRAM|URETHROGRAM",
    #     s
    # ):
    #     return "FLUORO_ANGIO_IR"

    # 8) Nuclear medicine (non-PET): radionuclide, radioisotope, SPECT, cardiac blood pool, V/Q, etc.
    if re.search(
        r"NUCLEAR MEDICINE|SPECT|RADIONUCLIDE|RADIOISOTOPE|SCINTI|TECHNETIUM|"
        r"MYOCARDIAL IMAGING|CARDIAC BLOOD POOL|PULMONARY PERFUSION|VENTILATION PERFUSION|"
        r"WHOLE BODY BONE|BONE IMAGING|MOLECULAR BREAST IMAGING|SALIVARY GLAND IMAGING|"
        r"THYROID IMAGING|THYROID UPTAKE",
        s
    ):
        return "NUCLEAR_MED"

    # 9) Plain X-ray / radiography
    if re.search(r"X-?RAY|PLAIN X-?RAY|RADIOGRAPH|RADIOLOGIC EXAMINATION|CHEST IMAGING", s):
        return "XRAY"

    # Generic "IMAGING" with no modality is ambiguous -> keep NaN
    if re.fullmatch(r"IMAGING|IMAGING OF .*|CHEST IMAGING", s) and not re.search(r"X-?RAY|RADIOGRAPH", s):
        return np.nan

    return np.nan

# Keep required columns
keep_cols = ["chai_patient_id", "report_date", "imaging_type_code", "imaging_type_name"]
keep_cols = [c for c in keep_cols if c in df.columns]
df_imaging = df[keep_cols].copy()

# Add scan_type_type
df_imaging["scan_type_type"] = df_imaging["imaging_type_name"].apply(classify_scan_type)

# Convert date to YYYY-MM-DD
df_imaging["report_date"] = to_iso_date(df_imaging["report_date"])
df_imaging = df_imaging[df_imaging["report_date"].notna()].copy()

# Quick check: how many unclassified?
# Quick check: how many unclassified?
n_total = len(df_imaging)
n_na = df_imaging["scan_type_type"].isna().sum()
print(f"[INFO] imaging rows={n_total:,}, unclassified scan_type_type={n_na:,} ({(n_na/max(n_total,1))*100:.2f}%)")

# Drop unspecified/unclassified (and anything we intentionally mapped to NaN: DXA, fluoro, non-breast ultrasound)
df_imaging = df_imaging[df_imaging["scan_type_type"].notna()].copy()

# Add a 1-based imaging sequence within each patient ordered by report_date.
# Rows from the same patient on the same date share the same sequence.
df_imaging = df_imaging.sort_values(
    ["chai_patient_id", "report_date", "imaging_type_code", "imaging_type_name"],
    ascending=[True, True, True, True],
    na_position="last",
).copy()
date_order = (
    df_imaging[["chai_patient_id", "report_date"]]
    .drop_duplicates()
    .sort_values(["chai_patient_id", "report_date"], ascending=[True, True])
    .copy()
)
date_order["sequence"] = date_order.groupby("chai_patient_id").cumcount() + 1
df_imaging = df_imaging.merge(date_order, on=["chai_patient_id", "report_date"], how="left")

write_csv_and_sqlite(df_imaging, "imaging")


# ## Imaging Trajectory Export (Integrated from `help_tools/export_imaging_trajectories.py`)
# 
# This section is integrated from `help_tools/export_imaging_trajectories.py` and runs immediately after the imaging extraction step.
# 
# What it does:
# - Uses the extracted `df_imaging` table (already cohort-filtered and scan-type normalized).
# - Collapses rows that share the same patient and imaging `sequence` into one visit label (scan types are joined with `+`).
# - Builds a patient-level trajectory string with month gaps between visits (for example: `6m -> CT(2021-07-01)`).
# - Exports two files to `data/scan_sequence/`: `imaging_trajectories.csv` and `imaging_trajectories.txt`.
# 

# In[9]:


from collections import defaultdict

DATE_FMT = "%Y-%m-%d"
SCAN_SEQUENCE_DIR = OUT_DIR.parent / "scan_sequence"
SCAN_SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

def month_gap(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

def normalize_scan_label(scan_types: set[str]) -> str:
    return "+".join(sorted(scan_types))

def read_visits_from_imaging_df(imaging_df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    visits_by_patient_and_sequence: dict[tuple[str, int], dict[str, object]] = {}

    req_cols = ["chai_patient_id", "report_date", "scan_type_type", "sequence"]
    missing = set(req_cols) - set(imaging_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in df_imaging: {sorted(missing)}")

    for row in imaging_df[req_cols].itertuples(index=False, name=None):
        patient_id, report_date_str, scan_type, sequence = row

        if pd.isna(patient_id) or pd.isna(report_date_str) or pd.isna(scan_type) or pd.isna(sequence):
            continue

        patient_id = str(patient_id).strip()
        report_date_str = str(report_date_str).strip()
        scan_type = str(scan_type).strip()

        if not patient_id or not report_date_str or not scan_type:
            continue

        report_date = pd.to_datetime(report_date_str, errors="coerce")
        if pd.isna(report_date):
            continue

        try:
            sequence = int(sequence)
        except Exception:
            continue

        key = (patient_id, sequence)
        if key not in visits_by_patient_and_sequence:
            visits_by_patient_and_sequence[key] = {
                "chai_patient_id": patient_id,
                "sequence": sequence,
                "report_date": report_date,
                "scan_types": set(),
            }

        visit = visits_by_patient_and_sequence[key]
        if report_date < visit["report_date"]:
            visit["report_date"] = report_date
        visit["scan_types"].add(scan_type)

    patient_visits: dict[str, list[dict[str, object]]] = defaultdict(list)
    for visit in visits_by_patient_and_sequence.values():
        patient_visits[visit["chai_patient_id"]].append(visit)

    for patient_id in patient_visits:
        patient_visits[patient_id].sort(key=lambda x: (x["sequence"], x["report_date"]))

    return patient_visits

def build_patient_rows(patient_visits: dict[str, list[dict[str, object]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for patient_id in sorted(patient_visits):
        visits = patient_visits[patient_id]
        if not visits:
            continue

        parts: list[str] = []
        previous_date = None

        for visit in visits:
            report_date = visit["report_date"]
            scan_label = normalize_scan_label(visit["scan_types"])
            visit_token = f"{scan_label}({report_date.strftime(DATE_FMT)})"

            if previous_date is None:
                parts.append(visit_token)
            else:
                parts.append(f"{month_gap(previous_date, report_date)}m -> {visit_token}")

            previous_date = report_date

        rows.append(
            {
                "chai_patient_id": patient_id,
                "trajectory": " -> ".join(parts),
                "n_sequences": str(len(visits)),
                "first_report_date": visits[0]["report_date"].strftime(DATE_FMT),
                "last_report_date": visits[-1]["report_date"].strftime(DATE_FMT),
            }
        )

    return rows

patient_visits = read_visits_from_imaging_df(df_imaging)
trajectory_rows = build_patient_rows(patient_visits)

TRAJECTORY_PATH = SCAN_SEQUENCE_DIR / "imaging_trajectories.csv"
TRAJECTORY_TXT_PATH = SCAN_SEQUENCE_DIR / "imaging_trajectories.txt"

pd.DataFrame(trajectory_rows).to_csv(TRAJECTORY_PATH, index=False)
with TRAJECTORY_TXT_PATH.open("w", encoding="utf-8") as f:
    for row in trajectory_rows:
        f.write(f"{row['chai_patient_id']}: {row['trajectory']}\n")

print(f"[OK] patients={len(trajectory_rows):,}")
print(f"[OK] csv={TRAJECTORY_PATH}")
print(f"[OK] txt={TRAJECTORY_TXT_PATH}")


# ## Trajectory Analysis (Integrated from `help_tools/trajectories_analysis.ipynb`)
# 
# This section is integrated from `help_tools/trajectories_analysis.ipynb` and now runs in the same notebook right after trajectory export.
# 
# Outputs produced in this integrated flow:
# - In-notebook cohort summaries by sequence count, long-gap status, and right-censoring bucket.
# - Text-based category tree for quick cohort navigation.
# - `data/raw_extract/trajectories_category.csv` for downstream modeling and reuse.
# 

# ## Trajectories Distribution
# 
# The next code cell reads `imaging_trajectories.csv` and uses `n_sequences` to split patients into two groups:
# - patients with exactly one scan sequence (`n_sequences == 1`)
# - patients with more than one scan sequence (`n_sequences > 1`)
# 
# It outputs a summary table with the unique patient count in each group.
# It also saves two reusable patient ID lists for later cells:
# - `patients_one_sequence`
# - `patients_more_than_one`
# 

# In[10]:


traj = pd.read_csv(TRAJECTORY_PATH)

traj["n_sequences"] = pd.to_numeric(traj["n_sequences"], errors="coerce")

patients_one_sequence = sorted(traj.loc[traj["n_sequences"] == 1, "chai_patient_id"].dropna().unique().tolist())
patients_more_than_one = sorted(traj.loc[traj["n_sequences"] > 1, "chai_patient_id"].dropna().unique().tolist())

n_patients_total = traj["chai_patient_id"].nunique()
n_patients_one_sequence = len(patients_one_sequence)
n_patients_more_than_one = len(patients_more_than_one)

summary = pd.DataFrame(
    {
        "group": ["total_patients", "n_sequences_eq_1", "n_sequences_gt_1"],
        "unique_patients": [n_patients_total, n_patients_one_sequence, n_patients_more_than_one],
    }
)

summary


# ## Trajectories Length
# 
# The next code cell focuses on the multi-sequence patient group (`n_sequences > 1`).
# It parses the `trajectory` text and extracts all interval values written as `Xm` between scan visits.
# 
# A manually adjustable threshold variable is used for the long-gap definition.
# For now it is set to `24` months, but it can be changed later if needed.
# 
# Each patient is then divided into one of two categories:
# - has at least one interval larger than the threshold
# - has no interval larger than the threshold
# 
# It outputs a summary table with the unique patient count in both categories.
# It also saves two reusable patient ID lists:
# - `patients_gap_gt_threshold`
# - `patients_gap_le_threshold`
# 

# In[11]:


gap_pattern = re.compile(r"(\d+)m")
GAP_THRESHOLD_MONTHS = 24

traj_multi = traj.loc[traj["n_sequences"] > 1].copy()
traj_multi["interval_months"] = traj_multi["trajectory"].fillna("").apply(
    lambda x: [int(v) for v in gap_pattern.findall(x)]
)
traj_multi["has_gap_gt_threshold"] = traj_multi["interval_months"].apply(
    lambda values: any(v > GAP_THRESHOLD_MONTHS for v in values)
)

patients_gap_gt_threshold = sorted(
    traj_multi.loc[traj_multi["has_gap_gt_threshold"], "chai_patient_id"].dropna().unique().tolist()
)
patients_gap_le_threshold = sorted(
    traj_multi.loc[~traj_multi["has_gap_gt_threshold"], "chai_patient_id"].dropna().unique().tolist()
)

trajectory_length_summary = pd.DataFrame(
    {
        "group": [
            f"n_sequences_gt_1_and_any_gap_gt_{GAP_THRESHOLD_MONTHS}m",
            f"n_sequences_gt_1_and_no_gap_gt_{GAP_THRESHOLD_MONTHS}m",
        ],
        "unique_patients": [len(patients_gap_gt_threshold), len(patients_gap_le_threshold)],
    }
)

trajectory_length_summary


# ## Right Censoring
# 
# The next code cell evaluates how far each patient's last observed scan is from a manually defined reference year.
# For now, the reference year is set to `2025`, and the code compares each patient's `last_report_date` to the end of that year.
# 
# It creates right-censoring buckets based on the month gap from the last scan to the reference year:
# - less than 1 year (`< 12m`)
# - 1 year
# - 2 year
# - 3 year
# - 4 year
# - 5 year
# - more than 5 year
# 
# It outputs three separate summary tables for these patient groups:
# - all patients with `n_sequences > 1`
# - patients with `n_sequences > 1` and no interval larger than the gap threshold
# - patients with `n_sequences > 1` and at least one interval larger than the gap threshold
# 

# In[12]:


THIS_YEAR = 2025
REFERENCE_DATE = pd.Timestamp(f"{THIS_YEAR}-12-31")

def months_from_last_scan(last_report_date, reference_date=REFERENCE_DATE):
    if pd.isna(last_report_date):
        return pd.NA
    last_date = pd.to_datetime(last_report_date, errors="coerce")
    if pd.isna(last_date):
        return pd.NA
    return (reference_date.year - last_date.year) * 12 + (reference_date.month - last_date.month)

def censor_bucket(months):
    if pd.isna(months):
        return pd.NA
    if months < 12:
        return "less_than_1_year"
    if months < 24:
        return "1_year"
    if months < 36:
        return "2_year"
    if months < 48:
        return "3_year"
    if months < 60:
        return "4_year"
    if months < 72:
        return "5_year"
    return "more_than_5_year"

def build_right_censoring_summary(df):
    tmp = df.copy()
    tmp["last_report_date"] = pd.to_datetime(tmp["last_report_date"], errors="coerce")
    tmp["months_from_last_scan"] = tmp["last_report_date"].apply(months_from_last_scan)
    tmp["right_censoring_bucket"] = tmp["months_from_last_scan"].apply(censor_bucket)
    bucket_order = [
        "less_than_1_year",
        "1_year",
        "2_year",
        "3_year",
        "4_year",
        "5_year",
        "more_than_5_year",
    ]
    summary = (
        tmp.groupby("right_censoring_bucket")["chai_patient_id"]
        .nunique()
        .reindex(bucket_order, fill_value=0)
        .reset_index(name="unique_patients")
    )
    return summary

right_censoring_all_gt1 = build_right_censoring_summary(traj_multi)
right_censoring_gt1_no_gap_gt_threshold = build_right_censoring_summary(
    traj_multi.loc[~traj_multi["has_gap_gt_threshold"]]
)
right_censoring_gt1_any_gap_gt_threshold = build_right_censoring_summary(
    traj_multi.loc[traj_multi["has_gap_gt_threshold"]]
)

right_censoring_all_gt1, right_censoring_gt1_no_gap_gt_threshold, right_censoring_gt1_any_gap_gt_threshold


# ## Category Tree
# 
# The next code cell builds an indented text tree of the current patient categories.
# Each node includes a count in parentheses so the hierarchy can be read like a cohort flow summary.
# It starts from all patients, then splits into `n_sequences == 1` and `n_sequences > 1`, and then further splits the multi-sequence group by gap-threshold status and right-censoring buckets.
# 

# In[13]:


def summary_to_count_map(df):
    return dict(zip(df["right_censoring_bucket"], df["unique_patients"]))

rc_no_gap_counts = summary_to_count_map(right_censoring_gt1_no_gap_gt_threshold)
rc_any_gap_counts = summary_to_count_map(right_censoring_gt1_any_gap_gt_threshold)

category_tree_lines = [
    f"All patients ({n_patients_total})",
    f"  n_sequences == 1 ({n_patients_one_sequence})",
    f"  n_sequences > 1 ({n_patients_more_than_one})",
    f"    no gap > {GAP_THRESHOLD_MONTHS}m ({len(patients_gap_le_threshold)})",
    f"      less_than_1_year ({rc_no_gap_counts.get('less_than_1_year', 0)})",
    f"      1_year ({rc_no_gap_counts.get('1_year', 0)})",
    f"      2_year ({rc_no_gap_counts.get('2_year', 0)})",
    f"      3_year ({rc_no_gap_counts.get('3_year', 0)})",
    f"      4_year ({rc_no_gap_counts.get('4_year', 0)})",
    f"      5_year ({rc_no_gap_counts.get('5_year', 0)})",
    f"      more_than_5_year ({rc_no_gap_counts.get('more_than_5_year', 0)})",
    f"    any gap > {GAP_THRESHOLD_MONTHS}m ({len(patients_gap_gt_threshold)})",
    f"      less_than_1_year ({rc_any_gap_counts.get('less_than_1_year', 0)})",
    f"      1_year ({rc_any_gap_counts.get('1_year', 0)})",
    f"      2_year ({rc_any_gap_counts.get('2_year', 0)})",
    f"      3_year ({rc_any_gap_counts.get('3_year', 0)})",
    f"      4_year ({rc_any_gap_counts.get('4_year', 0)})",
    f"      5_year ({rc_any_gap_counts.get('5_year', 0)})",
    f"      more_than_5_year ({rc_any_gap_counts.get('more_than_5_year', 0)})",
]

category_tree_text = "\n".join(category_tree_lines)
print(category_tree_text)


# ## Category Export
# 
# The next code cell creates a patient-level category table with one row per patient.
# It uses boolean columns to mark which category each patient belongs to, including sequence group, gap-threshold group, and right-censoring bucket group.
# The result is saved to `data/raw_extract/trajectories_category.csv` for later reuse outside the notebook.
# 

# In[14]:


category_output_path = OUT_DIR / "trajectories_category.csv"

patient_categories = traj[["chai_patient_id", "n_sequences", "first_report_date", "last_report_date"]].copy()
patient_categories["is_n_sequences_eq_1"] = patient_categories["n_sequences"] == 1
patient_categories["is_n_sequences_gt_1"] = patient_categories["n_sequences"] > 1

gap_flags = traj_multi[["chai_patient_id", "has_gap_gt_threshold"]].copy()
patient_categories = patient_categories.merge(gap_flags, on="chai_patient_id", how="left")
patient_categories["has_gap_gt_threshold"] = patient_categories["has_gap_gt_threshold"].fillna(False)
patient_categories["has_no_gap_gt_threshold"] = patient_categories["is_n_sequences_gt_1"] & (~patient_categories["has_gap_gt_threshold"])

patient_categories["last_report_date"] = pd.to_datetime(patient_categories["last_report_date"], errors="coerce")
patient_categories["months_from_last_scan"] = patient_categories["last_report_date"].apply(months_from_last_scan)
patient_categories["right_censoring_bucket"] = patient_categories["months_from_last_scan"].apply(censor_bucket)

bucket_order = [
    "less_than_1_year",
    "1_year",
    "2_year",
    "3_year",
    "4_year",
    "5_year",
    "more_than_5_year",
]

for bucket in bucket_order:
    patient_categories[f"is_right_censor_{bucket}"] = patient_categories["right_censoring_bucket"] == bucket

patient_categories["right_censor_all_gt_1"] = patient_categories["is_n_sequences_gt_1"]
patient_categories["right_censor_gt_1_no_gap_gt_threshold"] = patient_categories["has_no_gap_gt_threshold"]
patient_categories["right_censor_gt_1_any_gap_gt_threshold"] = patient_categories["is_n_sequences_gt_1"] & patient_categories["has_gap_gt_threshold"]

patient_categories.to_csv(category_output_path, index=False)
print(f"[OK] category file = {category_output_path.resolve()}")
patient_categories.head()


# ## Staging Time-series Table
# 
# Staging is extracted as a **time-series** table, so each patient can have multiple rows over time.
# 
# Output columns:
# - `chai_patient_id`
# - `stage_date`
# - `stage_group_name`
# 
# Stage processing:
# - Filter to the patient cohort.
# - Keep only rows with a valid `stage_date`.
# - Normalize raw stage group labels into coarse breast stage groups: `Stage 0`, `Stage I`, `Stage II`, `Stage III`, `Stage IV`.
# - If `stage_group_name` is missing or unusable but T/N/M values are available, derive stage group from TNM using the imported helper function.
# - Keep one row per patient-date-stage combination.

# In[15]:


p = DATA_DIR / "staging.parquet"
staging = pd.read_parquet(p)

# Filter to our patient cohort
staging = staging[staging["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Parse stage_date
staging["stage_date"] = pd.to_datetime(staging.get("stage_date"), errors="coerce")


def stage_group_coarse_or_none(stage_name):
    """
    Normalize raw stage labels into coarse breast stage groups.
    Returns one of: Stage 0, Stage I, Stage II, Stage III, Stage IV.
    """
    if pd.isna(stage_name):
        return None
    s = str(stage_name).upper().strip()

    exclusion_terms = [
        "FIGO", "ANN ARBOR", "ISS", "INSS",
        "LIMITED STAGE", "EXTENSIVE STAGE",
        "OCCULT", "UNKNOWN", "NOT APPLICABLE", "NOT REPORTED",
        "STAGE X", "STAGE 5",
        "TX", "RX",
    ]
    if any(t in s for t in exclusion_terms):
        return None

    if re.search(r"\bSTAGE\s*0\b", s):
        return "Stage 0"

    if re.search(r"\bSTAGE\s*(IA|IB|I|1)\b", s):
        return "Stage I"
    if re.search(r"\bSTAGE\s*(IIA|IIB|II|2)\b", s):
        return "Stage II"
    if re.search(r"\bSTAGE\s*(IIIA|IIIB|IIIC|III|3)\b", s):
        return "Stage III"
    if re.search(r"\bSTAGE\s*(IV|4)\b", s):
        return "Stage IV"

    return None


def simplify_t_stage(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"<NA>", "NA", "N/A", "UNKNOWN", "INVALID DATA", "NOT APPLICABLE", ""}:
        return None
    if re.search(r"\bP?TX\b", s):
        return "TX"
    if re.search(r"\bP?TIS\b", s):
        return "Tis"
    if re.search(r"\bP?T0\b", s):
        return "T0"
    if re.search(r"\bP?T1\b", s):
        return "T1"
    if re.search(r"\bP?T2\b", s):
        return "T2"
    if re.search(r"\bP?T3\b", s):
        return "T3"
    if re.search(r"\bP?T4\b", s):
        return "T4"
    return None


def simplify_n_stage(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"<NA>", "NA", "N/A", "UNKNOWN", "INVALID DATA", "NOT APPLICABLE", ""}:
        return None
    if re.search(r"\bP?NX\b", s):
        return "NX"
    if re.search(r"\bP?N1MI\b", s):
        return "N1mi"
    if re.search(r"\bP?N0\b", s):
        return "N0"
    if re.search(r"\bP?N1\b", s):
        return "N1"
    if re.search(r"\bP?N2\b", s):
        return "N2"
    if re.search(r"\bP?N3\b", s):
        return "N3"
    return None


def simplify_m_stage(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"<NA>", "NA", "N/A", "UNKNOWN", "INVALID DATA", "NOT APPLICABLE", ""}:
        return None
    if re.search(r"\bP?MX\b", s):
        return "MX"
    if re.search(r"\bP?M0\b", s):
        return "M0"
    if re.search(r"\bP?M1\b", s):
        return "M1"
    return None


def stage_group_from_tnm_or_none(t_raw, n_raw, m_raw):
    derived = raw_tnm_to_stage(t_raw, n_raw, m_raw, keep_stage0=True, detailed=False)
    mapping = {
        0: "Stage 0",
        1: "Stage I",
        2: "Stage II",
        3: "Stage III",
        4: "Stage IV",
    }
    return mapping.get(derived)

# Prefer normalized stage_group_name; if missing, derive from TNM
staging["stage_group_name_final"] = staging.get("stage_group_name").apply(stage_group_coarse_or_none)

# TNM simplified (guard for missing columns)
staging["t_stage_simple"] = staging["tstage_name"].apply(simplify_t_stage) if "tstage_name" in staging.columns else None
staging["n_stage_simple"] = staging["nstage_name"].apply(simplify_n_stage) if "nstage_name" in staging.columns else None
staging["m_stage_simple"] = staging["mstage_name"].apply(simplify_m_stage) if "mstage_name" in staging.columns else None

staging["stage_group_name_final"] = staging.apply(
    lambda row: row["stage_group_name_final"]
    if pd.notna(row["stage_group_name_final"])
    else stage_group_from_tnm_or_none(
        row.get("tstage_name"),
        row.get("nstage_name"),
        row.get("mstage_name"),
    ),
    axis=1,
)

# Keep only rows with a valid date AND a usable stage group
staging = staging[staging["stage_date"].notna()].copy()

staging = staging[staging["stage_group_name_final"].notna()].copy()

# Build time-series staging table
df_staging = staging[["chai_patient_id", "stage_date", "stage_group_name_final"]].copy()
df_staging = df_staging.rename(columns={"stage_group_name_final": "stage_group_name"})
df_staging["stage_date"] = to_iso_date(df_staging["stage_date"])
df_staging = df_staging[df_staging["stage_date"].notna()].copy()
df_staging = df_staging.drop_duplicates(subset=["chai_patient_id", "stage_date", "stage_group_name"])
df_staging = df_staging.sort_values(["chai_patient_id", "stage_date", "stage_group_name"], ascending=[True, True, True]).copy()

write_csv_and_sqlite(df_staging, "staging")


# ## Surgery Time-series Table
# 
# - Filter rows using `cohort_ids`.
# - Keep columns: `chai_patient_id`, `surgery_date`, `surgery_code`, `surgery_name`.
# - Convert `surgery_date` to `YYYY-MM-DD` format.
# - Drop rows where `surgery_date` is missing (or cannot be parsed).

# In[16]:


p = DATA_DIR / "surgery.parquet"
df = pd.read_parquet(p)

# Filter to our patient cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Keep required columns
keep_cols = [
    "chai_patient_id",
    "surgery_date",
    "surgery_code",
    "surgery_name",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_surgery = df[keep_cols].copy()

# Convert date to YYYY-MM-DD
df_surgery["surgery_date"] = to_iso_date(df_surgery["surgery_date"])

# Drop rows without surgery_date (not usable as time-series)
df_surgery = df_surgery[df_surgery["surgery_date"].notna()].copy()

write_csv_and_sqlite(df_surgery, "surgery")


# ## Tumor Exam Time-series Table
# 
# - Filter rows using `cohort_ids`.
# - Keep columns: `chai_patient_id`, `exam_date`, `exam_name_code`, `exam_name_name`.
# - Convert `exam_date` to `YYYY-MM-DD` format.
# - Drop rows where `exam_date` is missing (or cannot be parsed).

# In[17]:


p = DATA_DIR / "tumor_exam.parquet"
df = pd.read_parquet(p)

# Filter to our patient cohort
df = df[df["chai_patient_id"].astype(str).isin(cohort_ids)].copy()

# Keep required columns
keep_cols = [
    "chai_patient_id",
    "exam_date",
    "exam_name_code",
    "exam_name_name",
]
keep_cols = [c for c in keep_cols if c in df.columns]
df_tumor_exam = df[keep_cols].copy()

# Convert date to YYYY-MM-DD
df_tumor_exam["exam_date"] = to_iso_date(df_tumor_exam["exam_date"])

# Drop rows without exam_date (not usable as time-series)
df_tumor_exam = df_tumor_exam[df_tumor_exam["exam_date"].notna()].copy()

write_csv_and_sqlite(df_tumor_exam, "tumor_exam")

