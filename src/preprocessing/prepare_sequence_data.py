#!/usr/bin/env python
# coding: utf-8

# # Multitask Data Prepare
# 
# This notebook prepares five multitask datasets:
# - a static demographic table with one row per patient
# - an imaging time-series table with one row per patient/date/sequence
# - a staging time-series table with one row per patient/date/stage
# - a mastectomy surgery time-series table with one row per patient/date/code/name
# - a biomarker time-series table with one row per patient/date/biomarker/result
# 
# It also supports optional patient filtering using boolean columns from `trajectories_category.csv`.

# ## Setup
# 
# The next cell imports dependencies, resolves input and output folders, and defines helper functions used later in the notebook.
# This includes:
# - locating `raw_extract` and source parquet folders
# - parsing boolean filter columns
# - normalizing detailed stage labels
# - combining scan types within the same imaging sequence

# In[11]:


import os
from pathlib import Path
import re
import pandas as pd

from stage_utils import raw_tnm_to_stage

repo_root = Path(__file__).resolve().parents[2]
RAW_EXTRACT_DIR = Path(
    os.environ.get("RAW_EXTRACT_DIR", repo_root / "data" / "raw_extract")
).expanduser().resolve()
if not RAW_EXTRACT_DIR.exists():
    raise FileNotFoundError(f"Restricted raw extract directory not found: {RAW_EXTRACT_DIR}")

raw_data_value = os.environ.get("RAW_DATA_DIR")
SOURCE_DIR = Path(raw_data_value).expanduser().resolve() if raw_data_value else repo_root / "data" / "raw"
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Licensed Parquet directory not found: {SOURCE_DIR}. "
        "Place the authorized ConcertAI files in data/raw or set RAW_DATA_DIR."
    )

OUTPUT_DIR = Path(
    os.environ.get("MULTITASK_DATA_DIR", repo_root / "data" / "prepared" / "multitask")
).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CATEGORY_FILE = RAW_EXTRACT_DIR / "trajectories_category.csv"

def parse_bool_series(series: pd.Series) -> pd.Series:
    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna(False).map(lambda x: str(x).strip().lower() in truthy if not isinstance(x, bool) else x)

def normalize_stage_group_detailed(stage_name):
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
    if any(term in s for term in exclusion_terms):
        return None

    patterns = [
        ("0", [r"\bSTAGE\s*0\b"]),
        ("IA", [r"\bSTAGE\s*IA\b", r"\bSTAGE\s*1A\b"]),
        ("IB", [r"\bSTAGE\s*IB\b", r"\bSTAGE\s*1B\b"]),
        ("IIIA", [r"\bSTAGE\s*IIIA\b", r"\bSTAGE\s*3A\b"]),
        ("IIIB", [r"\bSTAGE\s*IIIB\b", r"\bSTAGE\s*3B\b"]),
        ("IIIC", [r"\bSTAGE\s*IIIC\b", r"\bSTAGE\s*3C\b"]),
        ("IIA", [r"\bSTAGE\s*IIA\b", r"\bSTAGE\s*2A\b"]),
        ("IIB", [r"\bSTAGE\s*IIB\b", r"\bSTAGE\s*2B\b"]),
        ("IV", [r"\bSTAGE\s*IV\b", r"\bSTAGE\s*4\b"]),
        ("III", [r"\bSTAGE\s*III\b", r"\bSTAGE\s*3\b"]),
        ("II", [r"\bSTAGE\s*II\b", r"\bSTAGE\s*2\b"]),
        ("I", [r"\bSTAGE\s*I\b", r"\bSTAGE\s*1\b"]),
    ]

    for label, regexes in patterns:
        if any(re.search(rx, s) for rx in regexes):
            return label

    return None


def stage_to_main(stage_name):
    if pd.isna(stage_name):
        return None
    s = str(stage_name).upper().strip()
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
    return None


def stage_to_binary(stage_name):
    main_stage = stage_to_main(stage_name)
    if main_stage is None:
        return None
    if main_stage == "IV":
        return "IV"
    return "0-III"

def combine_scan_types(values: pd.Series) -> str:
    unique_values = sorted({str(v).strip() for v in values.dropna() if str(v).strip()})
    return "+".join(unique_values)

SCAN_TYPE_ORDER = [
    "CT",
    "MRI",
    "PET",
    "ULTRASOUND",
    "XRAY",
    "MAMMO_TOMO",
    "NUCLEAR_MED",
]

def encode_scan_type_vector(values: pd.Series) -> str:
    present = {str(v).strip() for v in values.dropna() if str(v).strip()}
    return "[" + ",".join("1" if scan_type in present else "0" for scan_type in SCAN_TYPE_ORDER) + "]"

print("RAW_EXTRACT_DIR =", RAW_EXTRACT_DIR)
print("SOURCE_DIR      =", SOURCE_DIR)
print("OUTPUT_DIR      =", OUTPUT_DIR)
print("CATEGORY_FILE   =", DEFAULT_CATEGORY_FILE)


# ## Patient Selection
# 
# The next cell defines which patients are included.
# By default it keeps all patients from `patient.csv`.
# If you set `USE_CATEGORY_FILTER = True`, it will read `trajectories_category.csv` and keep only patients where all selected boolean columns are `True`.
# 
# Example use case:
# - multi-sequence patients
# - no gap above threshold
# - more than 5 years from the reference date

# In[12]:


USE_CATEGORY_FILTER = True
CATEGORY_FILE = DEFAULT_CATEGORY_FILE

patient_path = RAW_EXTRACT_DIR / "patient.csv"
patient_all = pd.read_csv(patient_path, dtype={"chai_patient_id": "string"})
selected_ids = set(patient_all["chai_patient_id"].dropna().astype(str).unique())

if USE_CATEGORY_FILTER:
    if not CATEGORY_FILE.exists():
        raise FileNotFoundError(f"Category filter requested but file not found: {CATEGORY_FILE}")

    categories = pd.read_csv(CATEGORY_FILE, dtype={"chai_patient_id": "string"})
    required_cols = [
        "is_n_sequences_gt_1",
        "has_no_gap_gt_threshold",
        "is_right_censor_3_year",
        "is_right_censor_4_year",
        "is_right_censor_5_year",
        "is_right_censor_more_than_5_year",
    ]
    missing = [col for col in required_cols if col not in categories.columns]
    if missing:
        raise KeyError(f"Missing category columns: {missing}")

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

print(f"Selected patients: {len(selected_ids):,}")


# ## Demographic Table
# 
# The next cell builds a static demographic table with one row per patient.
# It keeps the requested columns:
# - patient ID
# - gender
# - race
# - year of birth
# - date of death
# - age

# In[13]:


demographic = patient_all[patient_all["chai_patient_id"].astype(str).isin(selected_ids)].copy()
demographic = demographic[[
    "chai_patient_id",
    "gender",
    "race",
    "year_of_birth",
    "date_of_death",
    "age_years",
]].copy()
demographic = demographic.rename(columns={
    "chai_patient_id": "patient_id",
    "age_years": "age",
})
demographic = demographic.drop_duplicates(subset=["patient_id"]).sort_values("patient_id").copy()

demographic_path = OUTPUT_DIR / "demographic.csv"
demographic.to_csv(demographic_path, index=False)
print(f"[OK] demographic -> {demographic_path} rows={len(demographic):,}")
demographic.head()


# ## Imaging Time Series
# 
# The next cell builds the imaging time-series table.
# It keeps one row per patient, report date, and sequence.
# If the same patient has multiple scan types in the same sequence, those scan types are combined into one label such as `CT+MRI`.
# If repeated rows have the same scan type, the combined label stays unique, for example `CT` rather than `CT+CT`.
# 
# It also adds `scan_type_vector`, a multi-hot vector stored as a string.
# The fixed vector order is `[CT, MRI, PET, ULTRASOUND, XRAY, MAMMO_TOMO, NUCLEAR_MED]`.
# For example, `[0,0,0,1,0,1,0]` means that at that timestamp the patient had `ULTRASOUND` and `MAMMO_TOMO`.

# In[14]:


imaging_path = RAW_EXTRACT_DIR / "imaging.csv"
imaging = pd.read_csv(imaging_path, dtype={"chai_patient_id": "string"})
imaging = imaging[imaging["chai_patient_id"].astype(str).isin(selected_ids)].copy()

imaging["report_date"] = pd.to_datetime(imaging["report_date"], errors="coerce")
imaging["sequence"] = pd.to_numeric(imaging["sequence"], errors="coerce").astype("Int64")
imaging = imaging[imaging["report_date"].notna()].copy()
imaging = imaging[imaging["sequence"].notna()].copy()
imaging = imaging[imaging["scan_type_type"].notna()].copy()

imaging_out = (
    imaging.groupby(["chai_patient_id", "report_date", "sequence"], as_index=False)
    .agg(
        scan_type_type=("scan_type_type", combine_scan_types),
        scan_type_vector=("scan_type_type", encode_scan_type_vector),
    )
    .rename(columns={"chai_patient_id": "patient_id"})
    .sort_values(["patient_id", "report_date", "sequence"], ascending=[True, True, True])
    .copy()
)
imaging_out["report_date"] = imaging_out["report_date"].dt.strftime("%Y-%m-%d")

imaging_out_path = OUTPUT_DIR / "imaging_timeseries.csv"
imaging_out.to_csv(imaging_out_path, index=False)
print(f"[OK] imaging -> {imaging_out_path} rows={len(imaging_out):,}")
imaging_out.head()


# ## Staging Time Series
# 
# The next cell builds a detailed staging time-series table directly from the raw `staging.parquet` source.
# It prefers detailed stage-group labels such as `0`, `IA`, `IIB`, `IIIC`, and `IV`.
# If the raw stage group is missing or unusable, it falls back to detailed TNM-based stage conversion using the helper in `stage_utils.py`.

# In[15]:


staging_path = SOURCE_DIR / "staging.parquet"
staging = pd.read_parquet(staging_path)
staging = staging[staging["chai_patient_id"].astype(str).isin(selected_ids)].copy()

staging["stage_date"] = pd.to_datetime(staging.get("stage_date"), errors="coerce")
staging = staging[staging["stage_date"].notna()].copy()

staging["stage_group_name"] = staging.get("stage_group_name").apply(normalize_stage_group_detailed)
staging["stage_group_name"] = staging.apply(
    lambda row: row["stage_group_name"]
    if pd.notna(row["stage_group_name"])
    else raw_tnm_to_stage(
        row.get("tstage_name"),
        row.get("nstage_name"),
        row.get("mstage_name"),
        keep_stage0=True,
        detailed=True,
    ),
    axis=1,
)

staging = staging[staging["stage_group_name"].notna()].copy()
staging["stage_main"] = staging["stage_group_name"].apply(stage_to_main)
staging["stage_binary"] = staging["stage_group_name"].apply(stage_to_binary)
staging = staging[staging["stage_main"].notna() & staging["stage_binary"].notna()].copy()

staging_out = staging[["chai_patient_id", "stage_date", "stage_group_name", "stage_main", "stage_binary"]].copy()
staging_out = staging_out.rename(columns={"chai_patient_id": "patient_id"})
staging_out["stage_date"] = staging_out["stage_date"].dt.strftime("%Y-%m-%d")
staging_out = staging_out.drop_duplicates(subset=["patient_id", "stage_date", "stage_group_name"])
staging_out = staging_out.sort_values(["patient_id", "stage_date", "stage_group_name"], ascending=[True, True, True]).copy()

staging_out_path = OUTPUT_DIR / "staging_timeseries.csv"
staging_out.to_csv(staging_out_path, index=False)
print(f"[OK] staging -> {staging_out_path} rows={len(staging_out):,}")
staging_out.head()


# ## Mastectomy Time Series
# 
# The next cell builds a mastectomy-only surgery time-series table from `surgery.csv`.
# It keeps only rows where `surgery_name` contains `mastectomy` and writes one row per patient/date/code/name record.
# 
# The 29 matched `surgery_name` values are:
# - `Biopsy of mastectomy scar`
# - `Extended radical mastectomy`
# - `Extended simple mastectomy`
# - `Insertion of breast implant on same day of mastectomy (ie, immediate)`
# - `Insertion or replacement of breast implant on separate day from mastectomy`
# - `Mastectomy for gynecomastia`
# - `Mastectomy with preservation of skin and nipple with synchronous implant`
# - `Mastectomy, modified radical, including axillary lymph nodes, with or without pectoralis minor muscle, but excluding pectoralis major muscle`
# - `Mastectomy, partial (eg, lumpectomy, tylectomy, quadrantectomy, segmentectomy)`
# - `Mastectomy, partial (eg, lumpectomy, tylectomy, quadrantectomy, segmentectomy); with axillary lymphadenectomy`
# - `Mastectomy, radical, including pectoral muscles, axillary lymph nodes`
# - `Mastectomy, simple, complete`
# - `Mastectomy, subcutaneous`
# - `Modified radical mastectomy`
# - `Partial mastectomy`
# - `Partial mastectomy with axillary lymphadenectomy`
# - `Placement of radiotherapy afterloading expandable catheter (single or multichannel) into the breast for interstitial radioelement application following partial mastectomy, includes imaging guidance; concurrent with partial mastectomy (List separately in addition to code for primary procedure)`
# - `Placement of radiotherapy afterloading expandable catheter (single or multichannel) into the breast for interstitial radioelement application following partial mastectomy, includes imaging guidance; on date separate from partial mastectomy`
# - `Radical mastectomy`
# - `Radical mastectomy including axillary lymph nodes`
# - `Radical mastectomy of bilateral breasts`
# - `Simple mastectomy`
# - `Simple mastectomy of bilateral breasts`
# - `Simple mastectomy with axillary lymph node sampling`
# - `Skin sparing mastectomy`
# - `Subcutaneous mastectomy`
# - `Total mastectomy and axillary dissection`
# - `Total mastectomy and excision of part of pectoral muscles and chest wall`
# - `Vertical pattern skin reducing mastectomy and reconstruction of breast`
# 

# In[16]:


surgery_path = RAW_EXTRACT_DIR / "surgery.csv"
surgery = pd.read_csv(surgery_path, dtype={"chai_patient_id": "string"})
surgery = surgery[surgery["chai_patient_id"].astype(str).isin(selected_ids)].copy()

surgery["surgery_name"] = surgery["surgery_name"].fillna("").astype(str)
surgery = surgery[surgery["surgery_name"].str.contains("mastectomy", case=False, regex=False)].copy()
surgery["surgery_date"] = pd.to_datetime(surgery["surgery_date"], errors="coerce")
surgery["surgery_code"] = pd.to_numeric(surgery["surgery_code"], errors="coerce").astype("Int64")
surgery = surgery[surgery["surgery_date"].notna()].copy()

mastectomy_out = surgery[["chai_patient_id", "surgery_date", "surgery_code", "surgery_name"]].copy()
mastectomy_out = mastectomy_out.rename(columns={"chai_patient_id": "patient_id"})
mastectomy_out["surgery_date"] = mastectomy_out["surgery_date"].dt.strftime("%Y-%m-%d")
mastectomy_out = mastectomy_out.drop_duplicates(subset=["patient_id", "surgery_date", "surgery_code", "surgery_name"])
mastectomy_out = mastectomy_out.sort_values(["patient_id", "surgery_date", "surgery_code", "surgery_name"], ascending=[True, True, True, True]).copy()

mastectomy_out_path = OUTPUT_DIR / "mastectomy_timeseries.csv"
mastectomy_out.to_csv(mastectomy_out_path, index=False)
print(f"[OK] mastectomy -> {mastectomy_out_path} rows={len(mastectomy_out):,}")
mastectomy_out.head()


# ## Biomarker Time Series
# 
# The next cell builds a biomarker time-series table from `biomarker.csv`.
# It keeps one row per patient/report-date/biomarker/result record.
# Only three biomarker names are kept in this output, and only `Positive` / `Negative` results are retained:
# 
# The most common biomarker names in this extract include:
# - `ERBB2 (HER2)`
# - `ESR1 (ER)`
# - `PGR (PR)`
# 

# In[17]:


biomarker_path = RAW_EXTRACT_DIR / "biomarker.csv"
biomarker = pd.read_csv(biomarker_path, dtype={"chai_patient_id": "string"})
biomarker = biomarker[biomarker["chai_patient_id"].astype(str).isin(selected_ids)].copy()

if "biomarker_name_code" not in biomarker.columns:
    raise ValueError(
        "biomarker.csv is missing biomarker_name_code. "
        "Please rerun data_extract.ipynb to export biomarker_name_code first."
    )

biomarker = biomarker[biomarker["biomarker_name_name"].isin(["ERBB2", "ESR1", "PGR"])].copy()
biomarker["report_date"] = pd.to_datetime(biomarker["report_date"], errors="coerce")
biomarker = biomarker[biomarker["report_date"].notna()].copy()
biomarker = biomarker[biomarker["test_result_name"].notna()].copy()
biomarker = biomarker[biomarker["test_result_name"].isin(["Positive", "Negative"])].copy()

biomarker_out = biomarker[["chai_patient_id", "report_date", "biomarker_name_code", "test_result_name"]].copy()
biomarker_out = biomarker_out.rename(columns={
    "chai_patient_id": "patient_id",
    "test_result_name": "result_name",
})
biomarker_out["report_date"] = biomarker_out["report_date"].dt.strftime("%Y-%m-%d")
biomarker_out = biomarker_out.drop_duplicates(subset=["patient_id", "report_date", "biomarker_name_code", "result_name"])
biomarker_out = biomarker_out.sort_values(["patient_id", "report_date", "biomarker_name_code", "result_name"], ascending=[True, True, True, True]).copy()

biomarker_out_path = OUTPUT_DIR / "biomarker_timeseries.csv"
biomarker_out.to_csv(biomarker_out_path, index=False)
print(f"[OK] biomarker -> {biomarker_out_path} rows={len(biomarker_out):,}")
biomarker_out.head()


# ## Output Summary
# 
# The next cell gives a quick row-count summary of the multitask outputs and confirms where they were written.

# In[18]:


output_summary = pd.DataFrame(
    {
        "table": ["demographic", "imaging_timeseries", "staging_timeseries", "mastectomy_timeseries", "biomarker_timeseries"],
        "rows": [len(demographic), len(imaging_out), len(staging_out), len(mastectomy_out), len(biomarker_out)],
        "path": [str(demographic_path), str(imaging_out_path), str(staging_out_path), str(mastectomy_out_path), str(biomarker_out_path)],
    }
)
output_summary

