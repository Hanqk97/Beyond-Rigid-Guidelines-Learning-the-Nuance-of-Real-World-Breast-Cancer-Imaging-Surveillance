#!/usr/bin/env python
# coding: utf-8

# # LSTM Multitask Comparison For Next-Scan Prediction With Data-Hyperband
# 
# This notebook tunes the LSTM multitask model with data-Hyperband separately for each DS-family experiment setting. Early rungs train on smaller patient-wise fractions of the training set, all rungs are scored on the full validation set, and the final selected model is evaluated once on the held-out test set.
# 

# ## Setup
# 
# The next cell imports libraries, resolves prepared multitask tables, defines DS-family experiments, and creates a timestamped output folder.
# 

# In[ ]:


import os
import copy
import io
import contextlib
import tempfile
from datetime import datetime
from pathlib import Path
import random
import re
import json
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_ml4h"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

with contextlib.redirect_stderr(io.StringIO()):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import auc, f1_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
CODE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "hopper_cluster" else SCRIPT_DIR

def resolve_data_dir():
    cwd = Path.cwd().resolve()
    candidates = []
    env_data_dir = os.environ.get("MULTITASK_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser().resolve())
    candidates = [
        *candidates,
        SCRIPT_DIR / "data" / "multitask",
        SCRIPT_DIR / "data",
        SCRIPT_DIR / "data" / "prepared" / "multitask",
        CODE_DIR / "data" / "prepared" / "multitask",
        cwd / "data",
        cwd / "data" / "prepared" / "multitask",
        cwd / "Code" / "data" / "prepared" / "multitask",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find prepared multitask data directory from cwd={cwd}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def resolve_runtime_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), False
    try:
        # Probe a tiny CUDA op so unsupported GPU architectures fail fast and cleanly.
        _ = torch.tensor([0.0], device="cuda")
        _ = _ + 1.0
        torch.cuda.synchronize()
        return torch.device("cuda"), True
    except Exception as exc:
        print(f"CUDA unavailable for this runtime ({exc}); falling back to CPU.")
        return torch.device("cpu"), False

DEVICE, USE_AMP = resolve_runtime_device()
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def print_runtime_banner():
    print("=" * 80)
    print("RUNTIME DEVICE CHECK")
    print("torch_version:", torch.__version__)
    print("torch_cuda_version:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("selected_device:", DEVICE)
    print("use_amp:", USE_AMP)
    if torch.cuda.is_available():
        print("cuda_device_count:", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device_{idx}_name:", torch.cuda.get_device_name(idx))
    print("=" * 80)

print_runtime_banner()

DATA_DIR = resolve_data_dir()

FEATURE_PATHS = {
    "demographic": DATA_DIR / "demographic.csv",
    "imaging_timeseries": DATA_DIR / "imaging_timeseries.csv",
    "staging_timeseries": DATA_DIR / "staging_timeseries.csv",
    "mastectomy_timeseries": DATA_DIR / "mastectomy_timeseries.csv",
    "biomarker_timeseries": DATA_DIR / "biomarker_timeseries.csv",
}

SELECTED_EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "").strip()
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_EXPERIMENT_TAG = re.sub(r"[^A-Za-z0-9_.-]+", "_", SELECTED_EXPERIMENT_NAME or "ALL")
RESULTS_ROOT = Path(os.environ.get("MODEL_RESULTS_DIR", SCRIPT_DIR.parents[1] / "restricted_artifacts" / "results")).expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / f"lstm_tree_multitask_compare_next_scan_data_hyperband_normalized_score_{RUN_TIMESTAMP}_{RUN_EXPERIMENT_TAG}_pid{os.getpid()}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "report.txt"
REPORT_PATH.write_text("", encoding="utf-8")

CORE_INPUT_TABLES = ["demographic", "staging_timeseries", "imaging_timeseries"]
LABEL_SOURCE_TABLES = ["demographic", "staging_timeseries", "imaging_timeseries"]
# Naming: D=demographic, S=staging, I=imaging, M=mastectomy, B=biomarker
# Note: DSI appeared twice in the request; we keep unique experiment names.
EXPERIMENTS = [
    {"name": "DS", "feature_tables": ["demographic", "staging_timeseries"]},
    {"name": "DSI", "feature_tables": ["demographic", "staging_timeseries", "imaging_timeseries"]},
    {"name": "DSM", "feature_tables": ["demographic", "staging_timeseries", "mastectomy_timeseries"]},
    {"name": "DSB", "feature_tables": ["demographic", "staging_timeseries", "biomarker_timeseries"]},
    {"name": "DSIM", "feature_tables": ["demographic", "staging_timeseries", "imaging_timeseries", "mastectomy_timeseries"]},
    {"name": "DSIB", "feature_tables": ["demographic", "staging_timeseries", "imaging_timeseries", "biomarker_timeseries"]},
    {"name": "DSMB", "feature_tables": ["demographic", "staging_timeseries", "mastectomy_timeseries", "biomarker_timeseries"]},
    {"name": "DSIMB", "feature_tables": ["demographic", "staging_timeseries", "imaging_timeseries", "mastectomy_timeseries", "biomarker_timeseries"]},
]

if SELECTED_EXPERIMENT_NAME:
    selected_experiments = [exp for exp in EXPERIMENTS if exp["name"] == SELECTED_EXPERIMENT_NAME]
    if not selected_experiments:
        valid_names = ", ".join(exp["name"] for exp in EXPERIMENTS)
        raise ValueError(f"Unknown EXPERIMENT_NAME={SELECTED_EXPERIMENT_NAME!r}. Valid names: {valid_names}")
    EXPERIMENTS = selected_experiments
    print(f"[Parallel] EXPERIMENT_NAME={SELECTED_EXPERIMENT_NAME}; running only this experiment.")
else:
    print("[Parallel] EXPERIMENT_NAME not set; running all experiments serially.")

for name in LABEL_SOURCE_TABLES:
    if not FEATURE_PATHS[name].exists():
        raise FileNotFoundError(f"Missing required label-source table: {FEATURE_PATHS[name]}")
for experiment in EXPERIMENTS:
    for name in experiment["feature_tables"]:
        if not FEATURE_PATHS[name].exists():
            raise FileNotFoundError(f"Missing required experiment table: {FEATURE_PATHS[name]}")

INTERVAL_TASK_LOSS = "smooth_l1"  # options: "l1", "mse", "smooth_l1"
INTERVAL_ACC_THRESHOLDS = [0.5, 1.0, 2.0]
SCAN_TYPE_NAMES = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
N_SCAN_TYPES = len(SCAN_TYPE_NAMES)
PAD_TOKEN = "<PAD>"

BATCH_SIZE = 128
EPOCHS = int(os.environ.get("MODEL_EPOCHS", "20"))
LEARNING_RATE = 1e-3
PATIENCE = 5
DEFAULT_SCAN_THRESHOLD = 0.5

# Data-Hyperband tuning configuration. Each experiment setting gets its own best combination.
# Score is higher-is-better: scan_auc - (interval_mape / 100) / (1 + interval_mape / 100).
# Rung fractions are patient-wise fractions of the training split; validation stays full.
HYPERBAND_ENABLED = True
HYPERBAND_RUNG_TRAIN_FRACTIONS = [0.25, 0.50, 1.00]
HYPERBAND_EPOCHS_PER_TRIAL = EPOCHS
HYPERBAND_INITIAL_CONFIGS = int(os.environ.get("HYPERBAND_INITIAL_CONFIGS", "36"))
HYPERBAND_STAGE_SCHEDULE = {
    1: {
        "iterations": 4,  # 1 initial + 3 refinement rounds
        "next_stage_keep": 12,
        "refine_top_k": 12,
        "elite_keep": 2,
        "refresh_configs": 10,
    },
    2: {
        "iterations": 3,  # 1 initial + 2 refinement rounds
        "next_stage_keep": 4,
        "refine_top_k": 4,
        "elite_keep": 1,
        "refresh_configs": 3,
    },
    3: {
        "iterations": 3,  # 1 initial + 2 refinement rounds
        "next_stage_keep": 1,
        "refine_top_k": 2,
        "elite_keep": 1,
        "refresh_configs": 1,
    },
}

HYPERBAND_REFINE_EXPAND_RATIO = 0.20
# When building a refined numeric search range from the top candidates, expand
# the observed min/max range by 20% before clipping to the original search bounds.
# This keeps the next search focused near good candidates while still allowing
# some exploration around the current best region.
HYPERBAND_RANDOM_SEED = SEED
HYPERBAND_SCORE_NAME = "scan_auc_minus_normalized_interval_mape"

DEFAULT_HYPERPARAMS = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": 1e-5,
    "lstm_hidden_size": 64,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.15,
    "bidirectional": False,
    "shared_hidden_size": 256,
    "shared_dropout": 0.2,
    "stage_embedding_dim": 8,
    "gender_embedding_dim": 8,
    "mastectomy_embedding_dim": 16,
    "biomarker_code_embedding_dim": 16,
    "biomarker_result_embedding_dim": 8,
    "scan_loss_weight": 1.0,
    "interval_loss_weight": 1.0,
    "interval_task_loss": INTERVAL_TASK_LOSS,
}

# Hyperparameter search ranges used by sample_hyperparams() in the runner cell.
# "log_uniform" is sampled uniformly in log10 space; "uniform" is sampled linearly.
HYPERPARAM_SEARCH_SPACE = {
    "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
    "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 3e-3},
    "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-3},
    "lstm_hidden_size": {"type": "choice", "values": [64, 128, 256]},
    # "lstm_num_layers": {"type": "choice", "values": [1, 2]},
    # "lstm_dropout": {"type": "uniform", "low": 0.0, "high": 0.4},
    # "bidirectional": {"type": "choice", "values": [False, True]},
    "shared_hidden_size": {"type": "choice", "values": [128, 256, 512]},
    "shared_dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
    # "stage_embedding_dim": {"type": "choice", "values": [4, 8, 16]},
    # "gender_embedding_dim": {"type": "choice", "values": [4, 8, 16]},
    # "mastectomy_embedding_dim": {"type": "choice", "values": [8, 16, 32]},
    # "biomarker_code_embedding_dim": {"type": "choice", "values": [8, 16, 32]},
    # "biomarker_result_embedding_dim": {"type": "choice", "values": [4, 8, 16]},
    # "scan_loss_weight": {"type": "uniform", "low": 0.5, "high": 1.5},
    # "interval_loss_weight": {"type": "uniform", "low": 0.5, "high": 1.5},
    # "interval_task_loss": {"type": "choice", "values": ["smooth_l1", "l1", "mse"]},
}

WINDOWS_MULTIPROCESS_SAFE = os.name != "nt"
DATALOADER_NUM_WORKERS = 4 if (DEVICE.type == "cuda" and WINDOWS_MULTIPROCESS_SAFE) else 0
DATALOADER_PIN_MEMORY = DEVICE.type == "cuda"
DATALOADER_PERSISTENT_WORKERS = DATALOADER_NUM_WORKERS > 0
DATALOADER_PREFETCH_FACTOR = 2
print("DATALOADER_NUM_WORKERS:", DATALOADER_NUM_WORKERS)
print("DATALOADER_PIN_MEMORY:", DATALOADER_PIN_MEMORY)
print("DATALOADER_PERSISTENT_WORKERS:", DATALOADER_PERSISTENT_WORKERS)

def append_report_section(title, content):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n")
        f.write(content.rstrip() + "\n")


def save_dataframe_artifact(df, stem, title=None, index=False):
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    txt_path = OUTPUT_DIR / f"{stem}.txt"
    df.to_csv(csv_path, index=index)
    table_text = df.to_string(index=index)
    txt_path.write_text(table_text + "\n", encoding="utf-8")
    append_report_section(title or stem, table_text)
    return csv_path, txt_path


def save_figure_artifact(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    append_report_section(f"saved_figure::{stem}", str(png_path))
    return png_path


append_report_section("run_metadata", "\n".join([
    f"timestamp: {RUN_TIMESTAMP}",
    f"process_id: {os.getpid()}",
    f"selected_experiment_name: {SELECTED_EXPERIMENT_NAME or 'ALL'}",
    f"data_dir: {DATA_DIR}",
    f"output_dir: {OUTPUT_DIR}",
    f"seed: {SEED}",
]))

print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("label_source_tables:", LABEL_SOURCE_TABLES)
print("experiments:", EXPERIMENTS)
DATA_DIR





# ## Helper Functions
# 
# The next cell defines shared helpers for feature preparation, sequence encoding, training, prediction collection, and artifact export.
# 

# In[ ]:


def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


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


def simplify_stage_binary(stage_name):
    if pd.isna(stage_name):
        return "unknown"
    s = str(stage_name).strip().upper()
    if s == "IV" or s == "STAGE IV":
        return "IV"
    return "0-III"


def stringify_feature_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text if text else None


def parse_scan_vector(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = [float(v) for v in value]
    else:
        s = str(value).strip()
        s = s[1:-1] if s.startswith("[") and s.endswith("]") else s
        arr = [float(part.strip()) for part in s.split(",") if part.strip()]
    if len(arr) != N_SCAN_TYPES:
        raise ValueError(f"Expected scan vector length {N_SCAN_TYPES}, got {len(arr)} from {value}")
    return arr


def split_text_tokens(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("||") if part.strip()]


def build_vocab(values, add_pad=False, add_unk=False):
    vocab = []
    if add_pad:
        vocab.append(PAD_TOKEN)
    if add_unk:
        vocab.append("<UNK>")
    vocab.extend(sorted(set(values)))
    return {token: idx for idx, token in enumerate(vocab)}


def build_timeseries_text_snapshot(df, feature_name):
    date_cols = [c for c in df.columns if c.endswith("_date")]
    if not date_cols:
        raise ValueError(f"No *_date column found for time-series feature '{feature_name}'")
    date_col = date_cols[0]
    value_cols = [c for c in df.columns if c not in {"patient_id", date_col}]

    out = df[["patient_id", date_col] + value_cols].copy()
    out = out[out[date_col].notna()].copy()

    def row_to_text(row):
        tokens = []
        for col in value_cols:
            value = stringify_feature_value(row[col])
            if value is not None:
                tokens.append(f"{col}={value}")
        return " || ".join(sorted(set(tokens)))

    out[f"{feature_name}__text"] = out.apply(row_to_text, axis=1)
    out = out[out[f"{feature_name}__text"] != ""].copy()
    out = (
        out.groupby(["patient_id", date_col], as_index=False)[f"{feature_name}__text"]
        .agg(lambda values: " || ".join(sorted({v for v in values if v})))
        .sort_values([date_col, "patient_id"])
        .reset_index(drop=True)
    )
    return out, date_col


def build_static_text_snapshot(df, feature_name):
    value_cols = [c for c in df.columns if c != "patient_id"]
    out = df[["patient_id"] + value_cols].copy()

    def row_to_text(row):
        tokens = []
        for col in value_cols:
            value = stringify_feature_value(row[col])
            if value is not None:
                tokens.append(f"{col}={value}")
        return " || ".join(sorted(set(tokens)))

    out[f"{feature_name}__text"] = out.apply(row_to_text, axis=1)
    out = (
        out.groupby("patient_id", as_index=False)[f"{feature_name}__text"]
        .agg(lambda values: " || ".join(sorted({v for v in values if v})))
        .reset_index(drop=True)
    )
    return out


def build_text_vocab_from_examples(df, feature_col, source_col):
    tokens = set()
    for value in df[source_col]:
        if source_col == "history_text_features":
            for seq_value in value.get(feature_col, []):
                tokens.update(split_text_tokens(seq_value))
        else:
            tokens.update(split_text_tokens(value.get(feature_col, "")))
    return {token: idx for idx, token in enumerate(sorted(tokens))}


def encode_text_multihot(value, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in split_text_tokens(value):
        if token in vocab:
            vec[vocab[token]] = 1.0
    return vec


def safe_binary_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr, auc(fpr, tpr)


def safe_multilabel_micro_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = ~np.isnan(y_true.ravel())
    y_true_flat = y_true.ravel()[mask]
    y_prob_flat = y_prob.ravel()[mask]
    if len(np.unique(y_true_flat)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    return fpr, tpr, auc(fpr, tpr)


def safe_multilabel_macro_curve(y_true, y_prob):
    valid = [idx for idx in range(y_true.shape[1]) if len(np.unique(y_true[:, idx])) >= 2]
    if not valid:
        return None
    base = np.unique(np.concatenate([roc_curve(y_true[:, idx], y_prob[:, idx])[0] for idx in valid]))
    mean_tpr = np.zeros_like(base)
    auc_values = []
    for idx in valid:
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_prob[:, idx])
        mean_tpr += np.interp(base, fpr, tpr)
        auc_values.append(auc(fpr, tpr))
    mean_tpr /= len(valid)
    return base, mean_tpr, float(np.mean(auc_values))


def safe_multiclass_micro_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    y_bin = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    return fpr, tpr, auc(fpr, tpr)


def safe_multiclass_macro_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    class_indices = [idx for idx in range(y_prob.shape[1]) if np.sum(y_true == idx) > 0 and np.sum(y_true != idx) > 0]
    if len(class_indices) < 2:
        return None
    y_bin = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
    base = np.unique(np.concatenate([roc_curve(y_bin[:, idx], y_prob[:, idx])[0] for idx in class_indices]))
    mean_tpr = np.zeros_like(base)
    auc_values = []
    for idx in class_indices:
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        mean_tpr += np.interp(base, fpr, tpr)
        auc_values.append(auc(fpr, tpr))
    mean_tpr /= len(class_indices)
    return base, mean_tpr, float(np.mean(auc_values))


def tune_scan_thresholds(y_true, y_prob):
    thresholds = np.full(y_true.shape[1], DEFAULT_SCAN_THRESHOLD, dtype=np.float32)
    for class_idx in range(y_true.shape[1]):
        if np.unique(y_true[:, class_idx]).size < 2:
            continue
        best_threshold = DEFAULT_SCAN_THRESHOLD
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            pred = (y_prob[:, class_idx] >= threshold).astype(int)
            score = f1_score(y_true[:, class_idx], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[class_idx] = best_threshold
    return thresholds


def compute_scan_f1(pred, scan_thresholds):
    y_true = np.asarray(pred["y_scan"], dtype=int)
    y_pred = (np.asarray(pred["scan_probs"]) >= scan_thresholds).astype(int)
    return {
        "scan_micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "scan_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ## Experiment Functions
# 
# The next cell defines the data pipeline, dataset, model, training loop, prediction collection, and export helpers for one experiment setting.
# Imaging always provides labels and anchor timeline, while input features vary by experiment.
# Sequence inputs include scan vectors/gaps, stage, mastectomy, and biomarker code/result histories when enabled.
# 

# In[ ]:


SEQ_NONE_TOKEN = "<NONE>"


def normalize_thresholds(values):
    out = []
    for value in values:
        threshold = float(value)
        if threshold not in out:
            out.append(threshold)
    return out


def interval_task_criterion(loss_name=None):
    if loss_name is None:
        loss_name = INTERVAL_TASK_LOSS
    loss_name = str(loss_name).strip().lower()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown INTERVAL_TASK_LOSS={INTERVAL_TASK_LOSS!r}; expected l1, mse, or smooth_l1")


def compute_interval_metrics(pred, thresholds):
    true_months = np.asarray(pred["y_interval"], dtype=float)
    pred_months = np.asarray(pred["interval_pred"], dtype=float)
    abs_error = np.abs(pred_months - true_months)
    nonzero_mask = true_months != 0
    mape = (
        100.0 * np.mean(
            np.abs(
                (true_months[nonzero_mask] - pred_months[nonzero_mask])
                / true_months[nonzero_mask]
            )
        )
        if nonzero_mask.any()
        else np.nan
    )

    metrics = {
        "interval_mae": float(abs_error.mean()) if len(abs_error) else np.nan,
        "interval_rmse": float(np.sqrt(np.mean((pred_months - true_months) ** 2))) if len(abs_error) else np.nan,
        "interval_mape": mape,
    }
    for threshold in thresholds:
        key = f"acc@{float(threshold):g}m"
        metrics[key] = float((abs_error <= float(threshold)).mean()) if len(abs_error) else np.nan
    return metrics


def build_sequence_vocab(values):
    tokens = [PAD_TOKEN, "<UNK>", SEQ_NONE_TOKEN]
    observed = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    for token in observed:
        if token not in tokens:
            tokens.append(token)
    return {token: idx for idx, token in enumerate(tokens)}


def flatten_sequence_column(df, column_name):
    values = []
    for seq in df[column_name]:
        values.extend(list(seq))
    return values


def compute_month_offsets(event_dates, reference_dates):
    offsets = []
    for event_date, ref_date in zip(event_dates, reference_dates):
        if pd.isna(event_date) or pd.isna(ref_date):
            offsets.append(np.nan)
        else:
            offsets.append(float(months_between(event_date, ref_date)))
    return offsets


def attach_latest_prior_event(visit_df, event_df, event_date_col, value_cols, prefix):
    out = visit_df.sort_values(["report_date", "patient_id", "sequence"]).reset_index(drop=True).copy()
    if event_df.empty:
        out[f"{prefix}_event_date"] = pd.NaT
        for col in value_cols:
            out[f"{prefix}_{col}"] = np.nan
        out[f"{prefix}_months_before_scan"] = np.nan
        return out

    event_cols = ["patient_id", event_date_col] + value_cols
    event = event_df[event_cols].copy()
    event = event[event[event_date_col].notna()].copy()
    event = event.sort_values([event_date_col, "patient_id"] + value_cols).reset_index(drop=True)
    rename_map = {event_date_col: f"{prefix}_event_date"}
    rename_map.update({col: f"{prefix}_{col}" for col in value_cols})
    event = event.rename(columns=rename_map)

    out = pd.merge_asof(
        out,
        event,
        by="patient_id",
        left_on="report_date",
        right_on=f"{prefix}_event_date",
        direction="backward",
    )
    out[f"{prefix}_months_before_scan"] = compute_month_offsets(
        out[f"{prefix}_event_date"],
        out["report_date"],
    )
    return out


class NextScanDataset(Dataset):
    def __init__(
        self,
        df,
        stage_vocab,
        gender_vocab,
        interval_vocab,
        mastectomy_code_vocab,
        biomarker_code_vocab,
        biomarker_result_vocab,
        num_stats,
    ):
        self.df = df.reset_index(drop=True)
        self.stage_vocab = stage_vocab
        self.gender_vocab = gender_vocab
        self.interval_vocab = interval_vocab
        self.mastectomy_code_vocab = mastectomy_code_vocab
        self.biomarker_code_vocab = biomarker_code_vocab
        self.biomarker_result_vocab = biomarker_result_vocab
        self.num_stats = num_stats

    def __len__(self):
        return len(self.df)

    def _encode_sequence(self, seq_values, vocab, fallback=SEQ_NONE_TOKEN):
        unk_id = vocab.get("<UNK>", 0)
        default_value = vocab.get(fallback, unk_id)
        encoded = []
        for value in seq_values:
            token = str(value).strip() if pd.notna(value) and str(value).strip() else fallback
            encoded.append(vocab.get(token, unk_id if token != fallback else default_value))
        return encoded

    def _coerce_numeric_sequence(self, seq_values):
        out = []
        for value in seq_values:
            if pd.isna(value):
                out.append(0.0)
            else:
                out.append(float(value))
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        scan_vectors = [parse_scan_vector(token) for token in row["history_scan_vectors"]]
        gap_values = self._coerce_numeric_sequence(row["history_gap_months"])
        stage_ids = self._encode_sequence(row["history_stage_binary"], self.stage_vocab, fallback="unknown")
        stage_months_before_anchor_scan = self._coerce_numeric_sequence(row["history_stage_months_before_anchor_scan"])
        mastectomy_code_ids = self._encode_sequence(row["history_mastectomy_code"], self.mastectomy_code_vocab)
        mastectomy_months_before_anchor_scan = self._coerce_numeric_sequence(row["history_mastectomy_months_before_anchor_scan"])
        biomarker_code_ids = self._encode_sequence(row["history_biomarker_code"], self.biomarker_code_vocab)
        biomarker_result_ids = self._encode_sequence(row["history_biomarker_result"], self.biomarker_result_vocab)
        biomarker_months_before_anchor_scan = self._coerce_numeric_sequence(row["history_biomarker_months_before_anchor_scan"])

        gender_id = self.gender_vocab.get(str(row["gender"]), self.gender_vocab["<UNK>"])
        year_of_birth = float(row["year_of_birth"]) if pd.notna(row["year_of_birth"]) else self.num_stats["year_of_birth"]["mean"]
        age_at_history_end = float(row["age_at_history_end"]) if pd.notna(row["age_at_history_end"]) else self.num_stats["age_at_history_end"]["mean"]
        numeric_features = [
            (year_of_birth - self.num_stats["year_of_birth"]["mean"]) / self.num_stats["year_of_birth"]["std"],
            (age_at_history_end - self.num_stats["age_at_history_end"]["mean"]) / self.num_stats["age_at_history_end"]["std"],
        ]

        return {
            "patient_id": row["patient_id"],
            "scan_vectors": scan_vectors,
            "gap_values": gap_values,
            "stage_ids": stage_ids,
            "stage_months_before_anchor_scan": stage_months_before_anchor_scan,
            "mastectomy_code_ids": mastectomy_code_ids,
            "mastectomy_months_before_anchor_scan": mastectomy_months_before_anchor_scan,
            "biomarker_code_ids": biomarker_code_ids,
            "biomarker_result_ids": biomarker_result_ids,
            "biomarker_months_before_anchor_scan": biomarker_months_before_anchor_scan,
            "gender_id": gender_id,
            "numeric_features": numeric_features,
            "target_scan_vector": parse_scan_vector(row["target_scan_vector"]),
            "target_interval_months": float(row["target_interval_months"]),
        }


def make_collate_fn(stage_pad_id, mastectomy_pad_id, biomarker_code_pad_id, biomarker_result_pad_id, n_scan_types, num_feature_cols):
    def collate_batch(batch):
        lengths = [len(item["scan_vectors"]) for item in batch]
        max_len = max(lengths)

        scan_tensor = torch.zeros((len(batch), max_len, n_scan_types), dtype=torch.float32)
        gap_tensor = torch.zeros((len(batch), max_len), dtype=torch.float32)
        stage_tensor = torch.full((len(batch), max_len), stage_pad_id, dtype=torch.long)
        stage_recency_tensor = torch.zeros((len(batch), max_len), dtype=torch.float32)
        mastectomy_code_tensor = torch.full((len(batch), max_len), mastectomy_pad_id, dtype=torch.long)
        mastectomy_recency_tensor = torch.zeros((len(batch), max_len), dtype=torch.float32)
        biomarker_code_tensor = torch.full((len(batch), max_len), biomarker_code_pad_id, dtype=torch.long)
        biomarker_result_tensor = torch.full((len(batch), max_len), biomarker_result_pad_id, dtype=torch.long)
        biomarker_recency_tensor = torch.zeros((len(batch), max_len), dtype=torch.float32)

        gender_tensor = torch.zeros(len(batch), dtype=torch.long)
        numeric_tensor = torch.zeros((len(batch), len(num_feature_cols)), dtype=torch.float32)
        target_scan_tensor = torch.zeros((len(batch), n_scan_types), dtype=torch.float32)
        target_interval_tensor = torch.zeros(len(batch), dtype=torch.float32)

        for i, item in enumerate(batch):
            seq_len = len(item["scan_vectors"])
            scan_tensor[i, :seq_len] = torch.tensor(item["scan_vectors"], dtype=torch.float32)
            gap_tensor[i, :seq_len] = torch.tensor(item["gap_values"], dtype=torch.float32)
            stage_tensor[i, :seq_len] = torch.tensor(item["stage_ids"], dtype=torch.long)
            stage_recency_tensor[i, :seq_len] = torch.tensor(item["stage_months_before_anchor_scan"], dtype=torch.float32)
            mastectomy_code_tensor[i, :seq_len] = torch.tensor(item["mastectomy_code_ids"], dtype=torch.long)
            mastectomy_recency_tensor[i, :seq_len] = torch.tensor(item["mastectomy_months_before_anchor_scan"], dtype=torch.float32)
            biomarker_code_tensor[i, :seq_len] = torch.tensor(item["biomarker_code_ids"], dtype=torch.long)
            biomarker_result_tensor[i, :seq_len] = torch.tensor(item["biomarker_result_ids"], dtype=torch.long)
            biomarker_recency_tensor[i, :seq_len] = torch.tensor(item["biomarker_months_before_anchor_scan"], dtype=torch.float32)
            gender_tensor[i] = item["gender_id"]
            numeric_tensor[i] = torch.tensor(item["numeric_features"], dtype=torch.float32)
            target_scan_tensor[i] = torch.tensor(item["target_scan_vector"], dtype=torch.float32)
            target_interval_tensor[i] = float(item["target_interval_months"])

        return {
            "patient_ids": [item["patient_id"] for item in batch],
            "scan_vectors": scan_tensor,
            "gap_values": gap_tensor,
            "stage_ids": stage_tensor,
            "stage_months_before_anchor_scan": stage_recency_tensor,
            "mastectomy_code_ids": mastectomy_code_tensor,
            "mastectomy_months_before_anchor_scan": mastectomy_recency_tensor,
            "biomarker_code_ids": biomarker_code_tensor,
            "biomarker_result_ids": biomarker_result_tensor,
            "biomarker_months_before_anchor_scan": biomarker_recency_tensor,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "gender_ids": gender_tensor,
            "numeric_features": numeric_tensor,
            "target_scan_vectors": target_scan_tensor,
            "target_interval_months": target_interval_tensor,
        }
    return collate_batch



class MultitaskLSTM(nn.Module):
    def __init__(
        self,
        n_scan_types,
        n_stage_states,
        n_genders,
        n_interval_bins,
        n_mastectomy_codes,
        n_biomarker_codes,
        n_biomarker_results,
        stage_pad_id,
        mastectomy_pad_id,
        biomarker_code_pad_id,
        biomarker_result_pad_id,
        hyperparams=None,
    ):
        super().__init__()
        hp = dict(DEFAULT_HYPERPARAMS)
        if hyperparams:
            hp.update(hyperparams)

        stage_embedding_dim = int(hp["stage_embedding_dim"])
        gender_embedding_dim = int(hp["gender_embedding_dim"])
        mastectomy_embedding_dim = int(hp["mastectomy_embedding_dim"])
        biomarker_code_embedding_dim = int(hp["biomarker_code_embedding_dim"])
        biomarker_result_embedding_dim = int(hp["biomarker_result_embedding_dim"])
        lstm_hidden_size = int(hp["lstm_hidden_size"])
        lstm_num_layers = int(hp["lstm_num_layers"])
        lstm_dropout = float(hp["lstm_dropout"]) if lstm_num_layers > 1 else 0.0
        bidirectional = bool(hp["bidirectional"])
        shared_hidden_size = int(hp["shared_hidden_size"])
        shared_dropout = float(hp["shared_dropout"])

        self.stage_embed = nn.Embedding(n_stage_states, stage_embedding_dim, padding_idx=stage_pad_id)
        self.gender_embed = nn.Embedding(n_genders, gender_embedding_dim)
        self.mastectomy_code_embed = nn.Embedding(n_mastectomy_codes, mastectomy_embedding_dim, padding_idx=mastectomy_pad_id)
        self.biomarker_code_embed = nn.Embedding(n_biomarker_codes, biomarker_code_embedding_dim, padding_idx=biomarker_code_pad_id)
        self.biomarker_result_embed = nn.Embedding(n_biomarker_results, biomarker_result_embedding_dim, padding_idx=biomarker_result_pad_id)

        seq_input_dim = (
            n_scan_types + 1 + stage_embedding_dim + 1 + mastectomy_embedding_dim + 1
            + biomarker_code_embedding_dim + biomarker_result_embedding_dim + 1
        )
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        combined_dim = lstm_output_dim + gender_embedding_dim + 2
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, shared_hidden_size),
            nn.ReLU(),
            nn.Dropout(shared_dropout),
        )
        self.scan_head = nn.Linear(shared_hidden_size, n_scan_types)
        self.interval_head = nn.Linear(shared_hidden_size, 1)

    def forward(self, scan_vectors, gap_values, stage_ids, stage_months_before_anchor_scan, mastectomy_code_ids, mastectomy_months_before_anchor_scan, biomarker_code_ids, biomarker_result_ids, biomarker_months_before_anchor_scan, lengths, gender_ids, numeric_features):
        seq_input = torch.cat([
            scan_vectors,
            gap_values.unsqueeze(-1),
            self.stage_embed(stage_ids),
            stage_months_before_anchor_scan.unsqueeze(-1),
            self.mastectomy_code_embed(mastectomy_code_ids),
            mastectomy_months_before_anchor_scan.unsqueeze(-1),
            self.biomarker_code_embed(biomarker_code_ids),
            self.biomarker_result_embed(biomarker_result_ids),
            biomarker_months_before_anchor_scan.unsqueeze(-1),
        ], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(seq_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            seq_repr = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            seq_repr = hidden[-1]

        shared_input = torch.cat([seq_repr, self.gender_embed(gender_ids), numeric_features], dim=-1)
        shared_repr = self.shared(shared_input)
        interval_pred = nn.functional.softplus(self.interval_head(shared_repr)).squeeze(-1)
        return self.scan_head(shared_repr), interval_pred



def build_visit_df(experiment):
    input_feature_names = list(dict.fromkeys(experiment["feature_tables"]))
    active_feature_names = list(dict.fromkeys(LABEL_SOURCE_TABLES + input_feature_names))
    feature_frames = {}
    for name in active_feature_names:
        feature_frames[name] = pd.read_csv(FEATURE_PATHS[name], dtype={"patient_id": "string"})

    for df in feature_frames.values():
        for col in [c for c in df.columns if c.endswith("_date")]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if "sequence" in df.columns:
            df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce").astype("Int64")

    structured_input_tables = {name for name in input_feature_names if name in CORE_INPUT_TABLES}
    demographic = feature_frames["demographic"]
    imaging = feature_frames["imaging_timeseries"]
    staging = feature_frames["staging_timeseries"]

    demographic_model = demographic[["patient_id", "gender", "year_of_birth"]].copy()
    demographic_model["year_of_birth"] = pd.to_numeric(demographic_model["year_of_birth"], errors="coerce")

    staging_model = staging[["patient_id", "stage_date", "stage_group_name"]].copy()
    staging_model = staging_model[staging_model["stage_date"].notna()].copy()
    staging_model["stage_binary"] = staging_model["stage_group_name"].apply(simplify_stage_binary)
    staging_model = staging_model.sort_values(["stage_date", "patient_id", "stage_binary"]).reset_index(drop=True)

    imaging_model = imaging[["patient_id", "report_date", "sequence", "scan_type_vector"]].copy()
    imaging_model = imaging_model[imaging_model["report_date"].notna()].copy()
    imaging_model = imaging_model.sort_values(["report_date", "patient_id", "sequence"]).reset_index(drop=True)

    visit_df = imaging_model.copy()
    if "demographic" in structured_input_tables:
        visit_df = visit_df.merge(demographic_model, on="patient_id", how="left")
    else:
        visit_df["gender"] = "unknown"
        visit_df["year_of_birth"] = np.nan
    visit_df["report_year"] = visit_df["report_date"].dt.year
    visit_df["age_at_visit"] = visit_df["report_year"] - visit_df["year_of_birth"]

    if "staging_timeseries" in structured_input_tables:
        visit_df = attach_latest_prior_event(visit_df, staging_model, "stage_date", ["stage_binary"], "stage")
        visit_df["stage_stage_binary"] = visit_df["stage_stage_binary"].fillna("unknown")
    else:
        visit_df["stage_stage_binary"] = "unknown"
        visit_df["stage_months_before_anchor_scan"] = 0.0

    if "mastectomy_timeseries" in input_feature_names:
        mastectomy_model = feature_frames["mastectomy_timeseries"][["patient_id", "surgery_date", "surgery_code"]].copy()
        mastectomy_model = mastectomy_model[mastectomy_model["surgery_date"].notna()].copy()
        mastectomy_model["surgery_code"] = mastectomy_model["surgery_code"].astype("Int64").astype(str)
        mastectomy_model = mastectomy_model.sort_values(["surgery_date", "patient_id", "surgery_code"]).reset_index(drop=True)
        visit_df = attach_latest_prior_event(visit_df, mastectomy_model, "surgery_date", ["surgery_code"], "mastectomy")
        visit_df["mastectomy_surgery_code"] = visit_df["mastectomy_surgery_code"].fillna(SEQ_NONE_TOKEN)
    else:
        visit_df["mastectomy_surgery_code"] = SEQ_NONE_TOKEN
        visit_df["mastectomy_months_before_anchor_scan"] = 0.0

    if "biomarker_timeseries" in input_feature_names:
        biomarker_source = feature_frames["biomarker_timeseries"]
        required_biomarker_cols = {"patient_id", "report_date", "biomarker_name_code", "result_name"}
        missing_biomarker_cols = required_biomarker_cols - set(biomarker_source.columns)
        if missing_biomarker_cols:
            raise ValueError(
                "biomarker_timeseries is missing required code-based columns: "
                f"{sorted(missing_biomarker_cols)}. "
                "Please regenerate prepared/multitask biomarker_timeseries with biomarker_name_code."
            )
        biomarker_model = biomarker_source[["patient_id", "report_date", "biomarker_name_code", "result_name"]].copy()
        biomarker_model = biomarker_model[biomarker_model["report_date"].notna()].copy()
        biomarker_model["biomarker_name_code"] = biomarker_model["biomarker_name_code"].fillna(SEQ_NONE_TOKEN).astype(str)
        biomarker_model["result_name"] = biomarker_model["result_name"].fillna(SEQ_NONE_TOKEN).astype(str)
        biomarker_model = biomarker_model.sort_values(["report_date", "patient_id", "biomarker_name_code", "result_name"]).reset_index(drop=True)
        visit_df = attach_latest_prior_event(visit_df, biomarker_model, "report_date", ["biomarker_name_code", "result_name"], "biomarker")
        # Keep tensor key names unchanged for downstream notebook compatibility; values now come from code column.
        visit_df["biomarker_biomarker_name_code"] = visit_df["biomarker_biomarker_name_code"].fillna(SEQ_NONE_TOKEN)
        visit_df["biomarker_result_name"] = visit_df["biomarker_result_name"].fillna(SEQ_NONE_TOKEN)
    else:
        visit_df["biomarker_biomarker_name_code"] = SEQ_NONE_TOKEN
        visit_df["biomarker_result_name"] = SEQ_NONE_TOKEN
        visit_df["biomarker_months_before_anchor_scan"] = 0.0

    for col in ["stage_months_before_anchor_scan", "mastectomy_months_before_anchor_scan", "biomarker_months_before_anchor_scan"]:
        if col in visit_df.columns:
            visit_df[col] = pd.to_numeric(visit_df[col], errors="coerce").fillna(0.0)

    visit_df["_sort_report_date"] = pd.to_datetime(visit_df["report_date"], errors="coerce")
    visit_seq = visit_df["sequence"] if "sequence" in visit_df.columns else pd.Series(0, index=visit_df.index)
    visit_df["_sort_sequence"] = pd.to_numeric(visit_seq, errors="coerce").fillna(-1).astype("int64")
    visit_df = visit_df.sort_values(["patient_id", "_sort_report_date", "_sort_sequence"], kind="mergesort").drop(columns=["_sort_report_date", "_sort_sequence"]).reset_index(drop=True)
    input_flags = {
        "use_demographic_input": "demographic" in structured_input_tables,
        "use_stage_input": "staging_timeseries" in structured_input_tables,
        "use_imaging_input": "imaging_timeseries" in structured_input_tables,
        "use_mastectomy_input": "mastectomy_timeseries" in input_feature_names,
        "use_biomarker_input": "biomarker_timeseries" in input_feature_names,
    }
    return visit_df, input_flags


def build_examples(visit_df, input_flags):
    examples = []
    zero_scan_vector = str([0.0] * N_SCAN_TYPES)
    work_df = visit_df.copy()
    work_df["_sort_report_date"] = pd.to_datetime(work_df["report_date"], errors="coerce")
    work_seq = work_df["sequence"] if "sequence" in work_df.columns else pd.Series(0, index=work_df.index)
    work_df["_sort_sequence"] = pd.to_numeric(work_seq, errors="coerce").fillna(-1).astype("int64")
    work_df = work_df.sort_values(["patient_id", "_sort_report_date", "_sort_sequence"], kind="mergesort").drop(columns=["_sort_report_date", "_sort_sequence"]).reset_index(drop=True)
    for patient_id, group in work_df.groupby("patient_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        gender = group.loc[0, "gender"] if input_flags["use_demographic_input"] and "gender" in group.columns else "unknown"
        year_of_birth = group.loc[0, "year_of_birth"] if input_flags["use_demographic_input"] and "year_of_birth" in group.columns else np.nan

        for target_idx in range(1, len(group)):
            history = group.iloc[:target_idx].copy()
            target_row = group.iloc[target_idx]
            prev_row = group.iloc[target_idx - 1]

            actual_history_scan_vectors = history["scan_type_vector"].astype(str).tolist()
            actual_history_gap_months = []
            for hist_idx in range(len(history)):
                if hist_idx == 0:
                    actual_history_gap_months.append(0.0)
                else:
                    gap = months_between(history.iloc[hist_idx - 1]["report_date"], history.iloc[hist_idx]["report_date"])
                    actual_history_gap_months.append(float(gap))

            if input_flags["use_imaging_input"]:
                history_scan_vectors = actual_history_scan_vectors
                history_gap_months = actual_history_gap_months
            else:
                history_scan_vectors = [zero_scan_vector] * len(history)
                history_gap_months = [0.0] * len(history)

            anchor_date = history.iloc[-1]["report_date"]

            if input_flags["use_stage_input"]:
                history_stage_binary = history["stage_stage_binary"].fillna("unknown").astype(str).tolist()
                history_stage_months_before_anchor_scan = [
                    0.0 if pd.isna(event_date) else float(months_between(event_date, anchor_date))
                    for event_date in history["stage_event_date"]
                ]
            else:
                history_stage_binary = ["unknown"] * len(history)
                history_stage_months_before_anchor_scan = [0.0] * len(history)

            if input_flags["use_mastectomy_input"]:
                history_mastectomy_code = history["mastectomy_surgery_code"].fillna(SEQ_NONE_TOKEN).astype(str).tolist()
                history_mastectomy_months_before_anchor_scan = [
                    0.0 if pd.isna(event_date) else float(months_between(event_date, anchor_date))
                    for event_date in history["mastectomy_event_date"]
                ]
            else:
                history_mastectomy_code = [SEQ_NONE_TOKEN] * len(history)
                history_mastectomy_months_before_anchor_scan = [0.0] * len(history)

            if input_flags["use_biomarker_input"]:
                history_biomarker_code = history["biomarker_biomarker_name_code"].fillna(SEQ_NONE_TOKEN).astype(str).tolist()
                history_biomarker_result = history["biomarker_result_name"].fillna(SEQ_NONE_TOKEN).astype(str).tolist()
                history_biomarker_months_before_anchor_scan = [
                    0.0 if pd.isna(event_date) else float(months_between(event_date, anchor_date))
                    for event_date in history["biomarker_event_date"]
                ]
            else:
                history_biomarker_code = [SEQ_NONE_TOKEN] * len(history)
                history_biomarker_result = [SEQ_NONE_TOKEN] * len(history)
                history_biomarker_months_before_anchor_scan = [0.0] * len(history)

            target_gap_days = (target_row["report_date"] - prev_row["report_date"]).days
            target_gap_months = float(target_gap_days) / 30.4375
            examples.append(
                {
                    "patient_id": str(patient_id),
                    "history_scan_vectors": history_scan_vectors,
                    "history_gap_months": history_gap_months,
                    "raw_history_scan_vectors": actual_history_scan_vectors,
                    "raw_history_gap_months": actual_history_gap_months,
                    "history_stage_binary": history_stage_binary,
                    "history_stage_months_before_anchor_scan": history_stage_months_before_anchor_scan,
                    "history_mastectomy_code": history_mastectomy_code,
                    "history_mastectomy_months_before_anchor_scan": history_mastectomy_months_before_anchor_scan,
                    "history_biomarker_code": history_biomarker_code,
                    "history_biomarker_result": history_biomarker_result,
                    "history_biomarker_months_before_anchor_scan": history_biomarker_months_before_anchor_scan,
                    "target_scan_vector": str(target_row["scan_type_vector"]),
                    "target_interval_months": target_gap_months,
                    "gender": str(gender) if pd.notna(gender) else "unknown",
                    "year_of_birth": float(year_of_birth) if pd.notna(year_of_birth) else np.nan,
                    "age_at_history_end": float(history.iloc[-1]["age_at_visit"]) if input_flags["use_demographic_input"] and pd.notna(history.iloc[-1]["age_at_visit"]) else np.nan,
                }
            )
    return pd.DataFrame(examples)


def make_patient_splits(patient_ids):
    patient_ids = sorted(pd.Series(patient_ids).astype(str).unique())
    if len(patient_ids) < 3:
        raise ValueError(f"Need at least 3 patients to create train/val/test splits, got {len(patient_ids)}")
    train_patients, temp_patients = train_test_split(patient_ids, test_size=0.30, random_state=SEED)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=SEED)
    return {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }


def split_examples(examples_df, split_patients=None):
    if examples_df.empty:
        raise ValueError("No training examples were created from the selected features.")
    if split_patients is None:
        split_patients = make_patient_splits(examples_df["patient_id"].unique())

    train_examples = examples_df[examples_df["patient_id"].isin(split_patients["train"])].reset_index(drop=True)
    val_examples = examples_df[examples_df["patient_id"].isin(split_patients["val"])].reset_index(drop=True)
    test_examples = examples_df[examples_df["patient_id"].isin(split_patients["test"])].reset_index(drop=True)

    if len(train_examples) == 0 or len(val_examples) == 0 or len(test_examples) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}"
        )
    return train_examples, val_examples, test_examples


def subsample_train_examples_by_patient(train_examples, fraction=1.0, seed=SEED):
    fraction = float(fraction)
    if fraction >= 1.0:
        return train_examples.reset_index(drop=True)
    if fraction <= 0.0:
        raise ValueError(f"train fraction must be > 0, got {fraction}")

    patient_ids = np.array(sorted(train_examples["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(int(seed))
    shuffled_patients = patient_ids.copy()
    rng.shuffle(shuffled_patients)
    n_keep = max(1, int(np.ceil(len(shuffled_patients) * fraction)))
    keep_patients = set(shuffled_patients[:n_keep])
    return train_examples[
        train_examples["patient_id"].astype(str).isin(keep_patients)
    ].reset_index(drop=True)


def run_epoch(
    model,
    loader,
    criterion_scan,
    criterion_interval,
    optimizer=None,
    scaler=None,
    experiment_name="",
    phase="",
    scan_loss_weight=1.0,
    interval_loss_weight=1.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0
    n_batches = len(loader)
    log_every = max(1, n_batches // 5)
    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(loader, start=1):
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                scan_logits, interval_pred = model(
                    batch["scan_vectors"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["gap_values"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["stage_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["stage_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["mastectomy_code_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["mastectomy_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_code_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_result_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["lengths"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["gender_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["numeric_features"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                )

                target_scan_vectors = batch["target_scan_vectors"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY)
                target_interval_months = batch["target_interval_months"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY)
                scan_loss = criterion_scan(scan_logits, target_scan_vectors)
                interval_loss = criterion_interval(interval_pred, target_interval_months)
                loss = scan_loss_weight * scan_loss + interval_loss_weight * interval_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = batch["scan_vectors"].size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                running_loss = total_loss / max(total_count, 1)
                print(f"[{experiment_name}] {phase} batch {batch_idx}/{n_batches} running_loss={running_loss:.4f}")

    return total_loss / max(total_count, 1)


def collect_predictions(model, loader):
    model.eval()
    scan_probs_list = []
    interval_pred_list = []
    y_scan_list = []
    y_interval_list = []
    patient_ids = []

    n_batches = len(loader)
    log_every = max(1, n_batches // 5)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                scan_logits, interval_pred = model(
                    batch["scan_vectors"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["gap_values"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["stage_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["stage_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["mastectomy_code_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["mastectomy_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_code_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_result_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["biomarker_months_before_anchor_scan"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["lengths"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["gender_ids"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                    batch["numeric_features"].to(DEVICE, non_blocking=DATALOADER_PIN_MEMORY),
                )
            scan_probs_list.append(torch.sigmoid(scan_logits).cpu().numpy())
            interval_pred_list.append(interval_pred.cpu().numpy())
            y_scan_list.append(batch["target_scan_vectors"].cpu().numpy())
            y_interval_list.append(batch["target_interval_months"].cpu().numpy())
            patient_ids.extend([str(pid) for pid in batch["patient_ids"]])
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                print(f"[Predict] batch {batch_idx}/{n_batches}")

    return {
        "patient_ids": np.asarray(patient_ids, dtype=object),
        "scan_probs": np.vstack(scan_probs_list),
        "interval_pred": np.concatenate(interval_pred_list),
        "y_scan": np.vstack(y_scan_list),
        "y_interval": np.concatenate(y_interval_list),
    }


def build_prediction_detail_df(pred, interval_thresholds, scan_thresholds):
    true_interval = np.asarray(pred["y_interval"], dtype=float)
    pred_interval = np.asarray(pred["interval_pred"], dtype=float)
    abs_error = np.abs(pred_interval - true_interval)
    detail_df = pd.DataFrame(
        {
            "patient_id": pred["patient_ids"],
            "true_interval_months": true_interval,
            "pred_interval_months": pred_interval,
            "interval_error_months": pred_interval - true_interval,
            "interval_abs_error_months": abs_error,
        }
    )
    for threshold in interval_thresholds:
        threshold_txt = f"{float(threshold):g}"
        detail_df[f"within_{threshold_txt}m"] = abs_error <= float(threshold)

    for idx, scan_name in enumerate(SCAN_TYPE_NAMES):
        detail_df[f"y_scan_{scan_name}"] = pred["y_scan"][:, idx].astype(int)
        detail_df[f"p_scan_{scan_name}"] = pred["scan_probs"][:, idx]
        detail_df[f"pred_scan_{scan_name}"] = (pred["scan_probs"][:, idx] >= scan_thresholds[idx]).astype(int)
        detail_df[f"threshold_{scan_name}"] = float(scan_thresholds[idx])

    return detail_df


def scan_vector_to_label(scan_vec):
    values = parse_scan_vector(scan_vec)
    labels = [name for idx, name in enumerate(SCAN_TYPE_NAMES) if idx < len(values) and values[idx] >= 0.5]
    return "+".join(labels) if labels else "NONE"


def format_history_scan_trajectory(history_scan_vectors, history_gap_months):
    if not history_scan_vectors:
        return ""

    parts = []
    for idx, vec in enumerate(history_scan_vectors):
        label = scan_vector_to_label(vec)
        if idx == 0:
            parts.append(label)
        else:
            gap = 0.0
            if idx < len(history_gap_months):
                try:
                    gap = float(history_gap_months[idx])
                except Exception:
                    gap = 0.0
            gap_txt = f"{int(gap)}m" if float(gap).is_integer() else f"{gap:.2f}m"
            parts.append(f"{gap_txt} -> {label}")
    return " -> ".join(parts)


def format_event_history(values, recencies):
    if values is None:
        return ""
    out = []
    for idx, value in enumerate(values):
        vtxt = str(value).strip() if pd.notna(value) and str(value).strip() else SEQ_NONE_TOKEN
        if recencies is not None and idx < len(recencies):
            try:
                m = float(recencies[idx])
                mtxt = f"{int(m)}m" if m.is_integer() else f"{m:.2f}m"
            except Exception:
                mtxt = "NA"
            out.append(f"{vtxt}({mtxt})")
        else:
            out.append(vtxt)
    return " | ".join(out)


def build_incorrect_prediction_df(detail_df, examples_df, experiment_name, feature_tables, input_flags, interval_thresholds=None):
    if interval_thresholds is None:
        interval_thresholds = INTERVAL_ACC_THRESHOLDS
    examples = examples_df.reset_index(drop=True)
    if len(detail_df) != len(examples):
        raise ValueError(f"Detail/example length mismatch for {experiment_name}: detail={len(detail_df)} examples={len(examples)}")

    pred_cols = [f"pred_scan_{n}" for n in SCAN_TYPE_NAMES]
    true_cols = [f"y_scan_{n}" for n in SCAN_TYPE_NAMES]

    scan_incorrect = (detail_df[pred_cols].to_numpy().astype(int) != detail_df[true_cols].to_numpy().astype(int)).any(axis=1)
    interval_incorrect = detail_df["interval_abs_error_months"].to_numpy(dtype=float) > float(max(interval_thresholds))
    wrong_idx = np.where(scan_incorrect | interval_incorrect)[0]

    rows = []
    for idx in wrong_idx:
        d = detail_df.iloc[idx]
        ex = examples.iloc[idx]

        true_scan_types = [n for n in SCAN_TYPE_NAMES if int(d[f"y_scan_{n}"]) == 1]
        pred_scan_types = [n for n in SCAN_TYPE_NAMES if int(d[f"pred_scan_{n}"]) == 1]
        scan_mismatch = [
            f"{n}:{int(d[f'y_scan_{n}'])}->{int(d[f'pred_scan_{n}'])}"
            for n in SCAN_TYPE_NAMES
            if int(d[f"y_scan_{n}"]) != int(d[f"pred_scan_{n}"])
        ]

        row = {
            "experiment": experiment_name,
            "feature_tables_used": ", ".join(feature_tables),
            "patient_id": d["patient_id"],
            "true_interval_months": float(d["true_interval_months"]),
            "pred_interval_months": float(d["pred_interval_months"]),
            "interval_error_months": float(d["interval_error_months"]),
            "interval_abs_error_months": float(d["interval_abs_error_months"]),
            "interval_correct": float(d["interval_abs_error_months"]) <= float(max(interval_thresholds)),
            "true_scan_types": "+".join(true_scan_types) if true_scan_types else "NONE",
            "pred_scan_types": "+".join(pred_scan_types) if pred_scan_types else "NONE",
            "scan_mismatch_detail": " | ".join(scan_mismatch),
            "scan_history_trajectory": format_history_scan_trajectory(
                ex.get("raw_history_scan_vectors", ex.get("history_scan_vectors", [])),
                ex.get("raw_history_gap_months", ex.get("history_gap_months", [])),
            ),
            "history_length": len(ex.get("history_scan_vectors", [])),
            "used_demographic": bool(input_flags.get("use_demographic_input", False)),
            "used_stage": bool(input_flags.get("use_stage_input", False)),
            "used_imaging": bool(input_flags.get("use_imaging_input", False)),
            "used_mastectomy": bool(input_flags.get("use_mastectomy_input", False)),
            "used_biomarker": bool(input_flags.get("use_biomarker_input", False)),
        }

        if input_flags.get("use_demographic_input", False):
            row["gender"] = ex.get("gender", "")
            row["year_of_birth"] = ex.get("year_of_birth", "")
            row["age_at_history_end"] = ex.get("age_at_history_end", "")

        if input_flags.get("use_stage_input", False):
            row["stage_history"] = format_event_history(
                ex.get("history_stage_binary", []),
                ex.get("history_stage_months_before_anchor_scan", []),
            )

        if input_flags.get("use_mastectomy_input", False):
            row["mastectomy_history"] = format_event_history(
                ex.get("history_mastectomy_code", []),
                ex.get("history_mastectomy_months_before_anchor_scan", []),
            )

        if input_flags.get("use_biomarker_input", False):
            codes = ex.get("history_biomarker_code", [])
            results = ex.get("history_biomarker_result", [])
            rec = ex.get("history_biomarker_months_before_anchor_scan", [])
            combined = []
            for j in range(max(len(codes), len(results))):
                c = str(codes[j]).strip() if j < len(codes) and pd.notna(codes[j]) and str(codes[j]).strip() else SEQ_NONE_TOKEN
                r = str(results[j]).strip() if j < len(results) and pd.notna(results[j]) and str(results[j]).strip() else SEQ_NONE_TOKEN
                if j < len(rec):
                    try:
                        m = float(rec[j])
                        mtxt = f"{int(m)}m" if m.is_integer() else f"{m:.2f}m"
                    except Exception:
                        mtxt = "NA"
                else:
                    mtxt = "NA"
                combined.append(f"{c}/{r}({mtxt})")
            row["biomarker_history"] = " | ".join(combined)

        rows.append(row)

    return pd.DataFrame(rows)


def build_shap_feature_df(detail_df, examples_df, experiment_name, feature_tables, input_flags):
    examples = examples_df.reset_index(drop=True)
    if len(detail_df) != len(examples):
        raise ValueError(f"Detail/example length mismatch for {experiment_name}: detail={len(detail_df)} examples={len(examples)}")

    rows = []
    for idx in range(len(detail_df)):
        d = detail_df.iloc[idx]
        ex = examples.iloc[idx]

        raw_scan_vectors = ex.get("raw_history_scan_vectors", ex.get("history_scan_vectors", []))
        raw_gap_months = ex.get("raw_history_gap_months", ex.get("history_gap_months", []))
        parsed = [parse_scan_vector(v) for v in raw_scan_vectors]

        scan_counts = np.sum(np.asarray(parsed, dtype=float), axis=0) if len(parsed) > 0 else np.zeros(N_SCAN_TYPES, dtype=float)
        last_scan = np.asarray(parsed[-1], dtype=float) if len(parsed) > 0 else np.zeros(N_SCAN_TYPES, dtype=float)

        gap_vals = []
        for g in raw_gap_months:
            try:
                gap_vals.append(float(g))
            except Exception:
                continue

        row = {
            "experiment": experiment_name,
            "feature_tables_used": ", ".join(feature_tables),
            "patient_id": d["patient_id"],
            "history_length": len(raw_scan_vectors),
            "gap_mean": float(np.mean(gap_vals)) if len(gap_vals) > 0 else 0.0,
            "gap_max": float(np.max(gap_vals)) if len(gap_vals) > 0 else 0.0,
            "gap_last": float(gap_vals[-1]) if len(gap_vals) > 0 else 0.0,
            "true_interval_months": float(d["true_interval_months"]),
            "pred_interval_months": float(d["pred_interval_months"]),
            "interval_error_months": float(d["interval_error_months"]),
            "interval_abs_error_months": float(d["interval_abs_error_months"]),
            "scan_error_any": int(any(int(d[f"y_scan_{n}"]) != int(d[f"pred_scan_{n}"]) for n in SCAN_TYPE_NAMES)),
        }

        row["error_any"] = int(row["interval_error"] or row["scan_error_any"])

        for j, scan_name in enumerate(SCAN_TYPE_NAMES):
            row[f"scan_count_{scan_name}"] = float(scan_counts[j]) if j < len(scan_counts) else 0.0
            row[f"scan_last_{scan_name}"] = float(last_scan[j]) if j < len(last_scan) else 0.0
            row[f"p_scan_{scan_name}"] = float(d[f"p_scan_{scan_name}"])

        if input_flags.get("use_demographic_input", False):
            row["gender"] = ex.get("gender", "unknown")
            row["year_of_birth"] = float(ex.get("year_of_birth")) if pd.notna(ex.get("year_of_birth")) else np.nan
            row["age_at_history_end"] = float(ex.get("age_at_history_end")) if pd.notna(ex.get("age_at_history_end")) else np.nan

        if input_flags.get("use_stage_input", False):
            stage_hist = ex.get("history_stage_binary", [])
            stage_rec = ex.get("history_stage_months_before_anchor_scan", [])
            row["stage_last"] = str(stage_hist[-1]) if len(stage_hist) > 0 else "unknown"
            try:
                row["stage_last_recency_months"] = float(stage_rec[-1]) if len(stage_rec) > 0 else 0.0
            except Exception:
                row["stage_last_recency_months"] = 0.0

        if input_flags.get("use_mastectomy_input", False):
            m_hist = ex.get("history_mastectomy_code", [])
            m_rec = ex.get("history_mastectomy_months_before_anchor_scan", [])
            row["mastectomy_last_code"] = str(m_hist[-1]) if len(m_hist) > 0 else SEQ_NONE_TOKEN
            try:
                row["mastectomy_last_recency_months"] = float(m_rec[-1]) if len(m_rec) > 0 else 0.0
            except Exception:
                row["mastectomy_last_recency_months"] = 0.0

        if input_flags.get("use_biomarker_input", False):
            b_code = ex.get("history_biomarker_code", [])
            b_result = ex.get("history_biomarker_result", [])
            b_rec = ex.get("history_biomarker_months_before_anchor_scan", [])
            row["biomarker_last_code"] = str(b_code[-1]) if len(b_code) > 0 else SEQ_NONE_TOKEN
            row["biomarker_last_result"] = str(b_result[-1]) if len(b_result) > 0 else SEQ_NONE_TOKEN
            try:
                row["biomarker_last_recency_months"] = float(b_rec[-1]) if len(b_rec) > 0 else 0.0
            except Exception:
                row["biomarker_last_recency_months"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment(experiment, split_patients=None, hyperparams=None, train_fraction=1.0, train_subset_seed=SEED, max_epochs=None, evaluate_test=True, trial_id=None):
    hp = dict(DEFAULT_HYPERPARAMS)
    if hyperparams:
        hp.update(hyperparams)
    run_epochs = int(max_epochs if max_epochs is not None else HYPERBAND_EPOCHS_PER_TRIAL)
    run_label = experiment["name"] if trial_id is None else f"{experiment['name']}::{trial_id}"
    print(f"\n=== Experiment {run_label} started ===")
    print(f"[{run_label}] feature_tables={experiment['feature_tables']}")
    print(f"[{experiment['name']}] step=build_visit_df")
    visit_df, input_flags = build_visit_df(experiment)
    print(f"[{experiment['name']}] step=build_examples")
    examples_df = build_examples(visit_df, input_flags)
    print(f"[{experiment['name']}] step=split_examples")
    train_examples_full, val_examples, test_examples = split_examples(examples_df, split_patients=split_patients)
    train_examples = subsample_train_examples_by_patient(train_examples_full, fraction=train_fraction, seed=train_subset_seed)
    train_patients_full = train_examples_full["patient_id"].nunique()
    train_patients = train_examples["patient_id"].nunique()
    val_patients = val_examples["patient_id"].nunique()
    test_patients = test_examples["patient_id"].nunique()
    print(
        f"[{experiment['name']}] split: "
        f"train_examples={len(train_examples)}/{len(train_examples_full)} train_patients={train_patients}/{train_patients_full} train_fraction={float(train_fraction):.2f} | "
        f"val_examples={len(val_examples)} val_patients={val_patients} | "
        f"test_examples={len(test_examples)} test_patients={test_patients}"
    )

    stage_vocab = {PAD_TOKEN: 0, "unknown": 1, "0-III": 2, "IV": 3}
    gender_vocab = build_vocab(train_examples["gender"].fillna("unknown").astype(str).tolist(), add_unk=True)
    interval_thresholds = normalize_thresholds(INTERVAL_ACC_THRESHOLDS)
    mastectomy_code_vocab = build_sequence_vocab(flatten_sequence_column(train_examples, "history_mastectomy_code"))
    biomarker_code_vocab = build_sequence_vocab(flatten_sequence_column(train_examples, "history_biomarker_code"))
    biomarker_result_vocab = build_sequence_vocab(flatten_sequence_column(train_examples, "history_biomarker_result"))

    num_feature_cols = ["year_of_birth", "age_at_history_end"]
    num_stats = {}
    for col in num_feature_cols:
        mean_val = train_examples[col].astype(float).mean()
        std_val = train_examples[col].astype(float).std()
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        num_stats[col] = {"mean": float(mean_val), "std": float(std_val)}

    train_dataset = NextScanDataset(train_examples, stage_vocab, gender_vocab, None, mastectomy_code_vocab, biomarker_code_vocab, biomarker_result_vocab, num_stats)
    val_dataset = NextScanDataset(val_examples, stage_vocab, gender_vocab, None, mastectomy_code_vocab, biomarker_code_vocab, biomarker_result_vocab, num_stats)
    test_dataset = NextScanDataset(test_examples, stage_vocab, gender_vocab, None, mastectomy_code_vocab, biomarker_code_vocab, biomarker_result_vocab, num_stats)

    collate_fn = make_collate_fn(
        stage_vocab[PAD_TOKEN],
        mastectomy_code_vocab[PAD_TOKEN],
        biomarker_code_vocab[PAD_TOKEN],
        biomarker_result_vocab[PAD_TOKEN],
        N_SCAN_TYPES,
        num_feature_cols,
    )
    loader_kwargs = {
        "batch_size": int(hp["batch_size"]),
        "collate_fn": collate_fn,
        "num_workers": DATALOADER_NUM_WORKERS,
        "pin_memory": DATALOADER_PIN_MEMORY,
        "persistent_workers": DATALOADER_PERSISTENT_WORKERS,
    }
    if DATALOADER_NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    print(
        f"[{experiment['name']}] dataloaders: "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}"
    )

    model = MultitaskLSTM(
        n_scan_types=N_SCAN_TYPES,
        n_stage_states=len(stage_vocab),
        n_genders=len(gender_vocab),
        n_interval_bins=1,
        n_mastectomy_codes=len(mastectomy_code_vocab),
        n_biomarker_codes=len(biomarker_code_vocab),
        n_biomarker_results=len(biomarker_result_vocab),
        stage_pad_id=stage_vocab[PAD_TOKEN],
        mastectomy_pad_id=mastectomy_code_vocab[PAD_TOKEN],
        biomarker_code_pad_id=biomarker_code_vocab[PAD_TOKEN],
        biomarker_result_pad_id=biomarker_result_vocab[PAD_TOKEN],
        hyperparams=hp,
    ).to(DEVICE)

    criterion_scan = nn.BCEWithLogitsLoss()
    criterion_interval = interval_task_criterion(hp["interval_task_loss"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_state = None
    best_val_loss = float("inf")
    wait = 0
    history = []
    print(f"[{run_label}] step=train (epochs={run_epochs}, patience={PATIENCE})")

    for epoch in range(1, run_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion_scan,
            criterion_interval,
            optimizer=optimizer,
            scaler=scaler,
            experiment_name=run_label,
            phase=f'train-epoch-{epoch:02d}',
            scan_loss_weight=float(hp["scan_loss_weight"]),
            interval_loss_weight=float(hp["interval_loss_weight"]),
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion_scan,
            criterion_interval,
            optimizer=None,
            scaler=None,
            experiment_name=run_label,
            phase=f'val-epoch-{epoch:02d}',
            scan_loss_weight=float(hp["scan_loss_weight"]),
            interval_loss_weight=float(hp["interval_loss_weight"]),
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{run_label}] epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[{run_label}] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[{run_label}] step=val (collect_predictions)")
    val_pred = collect_predictions(model, val_loader)
    val_metrics = compute_task_metrics(val_pred, interval_thresholds)
    print(f"[{run_label}] val metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items() if isinstance(v, (int, float, np.floating)) and not pd.isna(v)]))

    test_pred = None
    test_metrics = None
    if evaluate_test:
        print(f"[{run_label}] step=test (collect_predictions)")
        test_pred = collect_predictions(model, test_loader)
        test_metrics = compute_task_metrics(test_pred, interval_thresholds)
        print(f"[{run_label}] test metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating)) and not pd.isna(v)]))
    print(f"=== Experiment {run_label} completed ===")

    return {
        "name": experiment["name"],
        "feature_tables": experiment["feature_tables"],
        "history": pd.DataFrame(history),
        "pred": test_pred,
        "test_metrics": test_metrics,
        "val_pred": val_pred,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "hyperparams": hp,
        "model": model,
        "test_loader": test_loader,
        "input_flags": input_flags,
        "use_demographic_input": input_flags["use_demographic_input"],
        "use_stage_input": input_flags["use_stage_input"],
        "use_imaging_input": input_flags["use_imaging_input"],
        "use_mastectomy_input": input_flags["use_mastectomy_input"],
        "use_biomarker_input": input_flags["use_biomarker_input"],
        "interval_thresholds": interval_thresholds,
        "mastectomy_code_vocab": mastectomy_code_vocab,
        "biomarker_code_vocab": biomarker_code_vocab,
        "biomarker_result_vocab": biomarker_result_vocab,
        "split_sizes": {
            "train_examples": len(train_examples),
            "train_examples_full": len(train_examples_full),
            "train_patients": train_patients,
            "train_patients_full": train_patients_full,
            "train_fraction": float(train_fraction),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
        },
        "test_examples_df": test_examples.reset_index(drop=True).copy(),
    }


def collect_predictions_with_override(model, loader, override_fn=None):
    model.eval()
    scan_probs_list = []
    interval_pred_list = []
    y_scan_list = []
    y_interval_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch_for_model = {
                "scan_vectors": batch["scan_vectors"].clone(),
                "gap_values": batch["gap_values"].clone(),
                "stage_ids": batch["stage_ids"].clone(),
                "stage_months_before_anchor_scan": batch["stage_months_before_anchor_scan"].clone(),
                "mastectomy_code_ids": batch["mastectomy_code_ids"].clone(),
                "mastectomy_months_before_anchor_scan": batch["mastectomy_months_before_anchor_scan"].clone(),
                "biomarker_code_ids": batch["biomarker_code_ids"].clone(),
                "biomarker_result_ids": batch["biomarker_result_ids"].clone(),
                "biomarker_months_before_anchor_scan": batch["biomarker_months_before_anchor_scan"].clone(),
                "lengths": batch["lengths"].clone(),
                "gender_ids": batch["gender_ids"].clone(),
                "numeric_features": batch["numeric_features"].clone(),
            }

            if override_fn is not None:
                batch_for_model = override_fn(batch_for_model)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                scan_logits, interval_pred = model(
                    batch_for_model["scan_vectors"].to(DEVICE),
                    batch_for_model["gap_values"].to(DEVICE),
                    batch_for_model["stage_ids"].to(DEVICE),
                    batch_for_model["stage_months_before_anchor_scan"].to(DEVICE),
                    batch_for_model["mastectomy_code_ids"].to(DEVICE),
                    batch_for_model["mastectomy_months_before_anchor_scan"].to(DEVICE),
                    batch_for_model["biomarker_code_ids"].to(DEVICE),
                    batch_for_model["biomarker_result_ids"].to(DEVICE),
                    batch_for_model["biomarker_months_before_anchor_scan"].to(DEVICE),
                    batch_for_model["lengths"].to(DEVICE),
                    batch_for_model["gender_ids"].to(DEVICE),
                    batch_for_model["numeric_features"].to(DEVICE),
                )

            scan_probs_list.append(torch.sigmoid(scan_logits).cpu().numpy())
            interval_pred_list.append(interval_pred.cpu().numpy())
            y_scan_list.append(batch["target_scan_vectors"].cpu().numpy())
            y_interval_list.append(batch["target_interval_months"].cpu().numpy())

    return {
        "scan_probs": np.vstack(scan_probs_list),
        "interval_pred": np.concatenate(interval_pred_list),
        "y_scan": np.vstack(y_scan_list),
        "y_interval": np.concatenate(y_interval_list),
    }


def compute_task_metrics(pred, thresholds):
    scan_micro_curve = safe_multilabel_micro_curve(pred["y_scan"], pred["scan_probs"])
    scan_macro_curve = safe_multilabel_macro_curve(pred["y_scan"], pred["scan_probs"])
    interval_metrics = compute_interval_metrics(pred, thresholds)
    return {
        "scan_micro_auc": np.nan if scan_micro_curve is None else scan_micro_curve[2],
        "scan_macro_auc": np.nan if scan_macro_curve is None else scan_macro_curve[2],
        **interval_metrics,
    }

def get_category_blocks(result):
    blocks = []
    if result["use_imaging_input"]:
        blocks.append(("imaging", "imaging"))
        blocks.append(("gap", "gap"))
    if result["use_stage_input"]:
        blocks.append(("stage", "stage"))
    if result["use_demographic_input"]:
        blocks.append(("demographic", "demographic"))
    if result["use_mastectomy_input"]:
        blocks.append(("mastectomy", "mastectomy"))
    if result["use_biomarker_input"]:
        blocks.append(("biomarker", "biomarker"))
    return blocks


def permute_category_block(batch_for_model, block_name):
    out = {k: v.clone() for k, v in batch_for_model.items()}
    perm = torch.randperm(out["scan_vectors"].shape[0])

    if block_name == "imaging":
        out["scan_vectors"] = out["scan_vectors"][perm]
    elif block_name == "gap":
        out["gap_values"] = out["gap_values"][perm]
    elif block_name == "stage":
        out["stage_ids"] = out["stage_ids"][perm]
        out["stage_months_before_anchor_scan"] = out["stage_months_before_anchor_scan"][perm]
    elif block_name == "demographic":
        out["gender_ids"] = out["gender_ids"][perm]
        out["numeric_features"] = out["numeric_features"][perm]
    elif block_name == "mastectomy":
        out["mastectomy_code_ids"] = out["mastectomy_code_ids"][perm]
        out["mastectomy_months_before_anchor_scan"] = out["mastectomy_months_before_anchor_scan"][perm]
    elif block_name == "biomarker":
        out["biomarker_code_ids"] = out["biomarker_code_ids"][perm]
        out["biomarker_result_ids"] = out["biomarker_result_ids"][perm]
        out["biomarker_months_before_anchor_scan"] = out["biomarker_months_before_anchor_scan"][perm]
    else:
        raise ValueError(f"Unknown category block: {block_name}")

    return out


def get_exact_feature_blocks(result):
    blocks = []
    if result["use_demographic_input"]:
        blocks.extend([
            ("gender", "gender"),
            ("year_of_birth", "year_of_birth"),
            ("age_at_history_end", "age_at_history_end"),
        ])
    if result["use_imaging_input"]:
        blocks.extend([
            ("scan_vectors", "scan_vectors"),
            ("gap_values", "gap_values"),
        ])
    if result["use_stage_input"]:
        blocks.extend([
            ("stage_ids", "stage_ids"),
            ("stage_months_before_anchor_scan", "stage_months_before_anchor_scan"),
        ])
    if result["use_mastectomy_input"]:
        blocks.extend([
            ("mastectomy_code_ids", "mastectomy_code_ids"),
            ("mastectomy_months_before_anchor_scan", "mastectomy_months_before_anchor_scan"),
        ])
    if result["use_biomarker_input"]:
        blocks.extend([
            ("biomarker_code_ids", "biomarker_code_ids"),
            ("biomarker_result_ids", "biomarker_result_ids"),
            ("biomarker_months_before_anchor_scan", "biomarker_months_before_anchor_scan"),
        ])
    return blocks


def permute_exact_feature(batch_for_model, feature_type):
    out = {k: v.clone() for k, v in batch_for_model.items()}
    perm = torch.randperm(out["scan_vectors"].shape[0])

    if feature_type == "gender":
        out["gender_ids"] = out["gender_ids"][perm]
    elif feature_type == "year_of_birth":
        out["numeric_features"][:, 0] = out["numeric_features"][perm, 0]
    elif feature_type == "age_at_history_end":
        out["numeric_features"][:, 1] = out["numeric_features"][perm, 1]
    elif feature_type == "scan_vectors":
        out["scan_vectors"] = out["scan_vectors"][perm]
    elif feature_type == "gap_values":
        out["gap_values"] = out["gap_values"][perm]
    elif feature_type == "stage_ids":
        out["stage_ids"] = out["stage_ids"][perm]
    elif feature_type == "stage_months_before_anchor_scan":
        out["stage_months_before_anchor_scan"] = out["stage_months_before_anchor_scan"][perm]
    elif feature_type == "mastectomy_code_ids":
        out["mastectomy_code_ids"] = out["mastectomy_code_ids"][perm]
    elif feature_type == "mastectomy_months_before_anchor_scan":
        out["mastectomy_months_before_anchor_scan"] = out["mastectomy_months_before_anchor_scan"][perm]
    elif feature_type == "biomarker_code_ids":
        out["biomarker_code_ids"] = out["biomarker_code_ids"][perm]
    elif feature_type == "biomarker_result_ids":
        out["biomarker_result_ids"] = out["biomarker_result_ids"][perm]
    elif feature_type == "biomarker_months_before_anchor_scan":
        out["biomarker_months_before_anchor_scan"] = out["biomarker_months_before_anchor_scan"][perm]
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    return out


def compute_category_contributions(result):
    base_pred = collect_predictions_with_override(result["model"], result["test_loader"], override_fn=None)
    base_metrics = compute_task_metrics(base_pred, INTERVAL_ACC_THRESHOLDS)

    rows = []
    for display_name, block_name in get_category_blocks(result):
        perm_pred = collect_predictions_with_override(
            result["model"],
            result["test_loader"],
            override_fn=lambda batch, bn=block_name: permute_category_block(batch, bn),
        )
        perm_metrics = compute_task_metrics(perm_pred, INTERVAL_ACC_THRESHOLDS)
        rows.append(
            {
                "feature_group": display_name,
                "scan_contribution": base_metrics["scan_micro_auc"] - perm_metrics["scan_micro_auc"],
                "interval_contribution": perm_metrics["interval_mae"] - base_metrics["interval_mae"],
            }
        )

    return pd.DataFrame(rows), base_metrics


def compute_exact_feature_contributions(result):
    base_pred = collect_predictions_with_override(result["model"], result["test_loader"], override_fn=None)
    base_metrics = compute_task_metrics(base_pred, INTERVAL_ACC_THRESHOLDS)

    rows = []
    for feature_name, feature_type in get_exact_feature_blocks(result):
        perm_pred = collect_predictions_with_override(
            result["model"],
            result["test_loader"],
            override_fn=lambda batch, ft=feature_type: permute_exact_feature(batch, ft),
        )
        perm_metrics = compute_task_metrics(perm_pred, INTERVAL_ACC_THRESHOLDS)
        rows.append(
            {
                "feature_name": feature_name,
                "feature_type": feature_type,
                "scan_contribution": base_metrics["scan_micro_auc"] - perm_metrics["scan_micro_auc"],
                "interval_contribution": perm_metrics["interval_mae"] - base_metrics["interval_mae"],
            }
        )

    return pd.DataFrame(rows), base_metrics





# ## Runtime Device Check
# 
# The next cell performs a final GPU sanity-check before running experiments. If CUDA fails, it automatically switches to CPU and prints a reminder.
# 

# In[ ]:


print("[Preflight] Runtime device check before experiments...")
if DEVICE.type == "cuda":
    try:
        _probe = torch.randn(128, 128, device=DEVICE)
        _ = _probe @ _probe.T
        torch.cuda.synchronize()
        print("[Preflight] GPU check passed. Continuing with CUDA.")
    except Exception as exc:
        print(f"[Preflight] GPU check failed ({exc}). Switching to CPU.")
        DEVICE = torch.device("cpu")
        USE_AMP = False
        DATALOADER_NUM_WORKERS = 0
        DATALOADER_PIN_MEMORY = False
        DATALOADER_PERSISTENT_WORKERS = False
        DATALOADER_PREFETCH_FACTOR = 2
else:
    print("[Preflight] CPU is already selected.")

print("[Preflight] Final DEVICE:", DEVICE)
print("[Preflight] Reminder: if GPU is not compatible, this notebook returns to CPU automatically.")


# ## Run Data-Hyperband Experiments
# 
# The runner keeps the original patient-level train/validation/test split. Hyperband only changes how much of the training split is used per rung: 25%, 50%, then 100% of training patients. Validation remains 100% for ranking configurations, and test is used only after the best configuration is selected.
# 

# In[ ]:


label_anchor_experiment = {"name": "label_anchor", "feature_tables": LABEL_SOURCE_TABLES}
print("[Runner] Building baseline examples for shared patient split...")
baseline_visit_df, baseline_input_flags = build_visit_df(label_anchor_experiment)
baseline_examples_df = build_examples(baseline_visit_df, baseline_input_flags)
shared_split_patients = make_patient_splits(baseline_examples_df["patient_id"].unique())
print(
    "[Runner] shared patient split: "
    f"train={len(shared_split_patients['train'])} "
    f"val={len(shared_split_patients['val'])} "
    f"test={len(shared_split_patients['test'])}"
)
append_report_section(
    "split_strategy",
    "Patient-level single split with fixed random seed.\n"
    f"train_patients={len(shared_split_patients['train'])}\n"
    f"val_patients={len(shared_split_patients['val'])}\n"
    f"test_patients={len(shared_split_patients['test'])}\n"
    "This notebook tunes each experiment setting with data-Hyperband on validation metrics. "
    "Each rung uses a patient-wise fraction of the training split while validation stays full; "
    "the best combination is evaluated once on the held-out test set.",
)


def sample_from_space(rng, spec):
    if spec["type"] == "choice":
        value = rng.choice(spec["values"])
        if isinstance(value, np.generic):
            value = value.item()
        return value
    if spec["type"] == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if spec["type"] == "log_uniform":
        return float(10 ** rng.uniform(np.log10(float(spec["low"])), np.log10(float(spec["high"]))))
    raise ValueError(f"Unknown search-space type: {spec['type']}")


def sample_hyperparams(rng, search_space=None):
    hp = dict(DEFAULT_HYPERPARAMS)

    if search_space is None:
        search_space = HYPERPARAM_SEARCH_SPACE

    for name, spec in search_space.items():
        hp[name] = sample_from_space(rng, spec)

    hp["batch_size"] = int(hp["batch_size"])
    hp["lstm_hidden_size"] = int(hp["lstm_hidden_size"])
    hp["lstm_num_layers"] = int(hp["lstm_num_layers"])
    hp["shared_hidden_size"] = int(hp["shared_hidden_size"])
    hp["stage_embedding_dim"] = int(hp["stage_embedding_dim"])
    hp["gender_embedding_dim"] = int(hp["gender_embedding_dim"])
    hp["mastectomy_embedding_dim"] = int(hp["mastectomy_embedding_dim"])
    hp["biomarker_code_embedding_dim"] = int(hp["biomarker_code_embedding_dim"])
    hp["biomarker_result_embedding_dim"] = int(hp["biomarker_result_embedding_dim"])
    hp["bidirectional"] = bool(hp["bidirectional"])
    hp["interval_task_loss"] = str(hp["interval_task_loss"])
    if hp["lstm_num_layers"] == 1:
        hp["lstm_dropout"] = 0.0
    return hp


def make_refined_search_space(top_rows, base_space=None, expand_ratio=0.20):
    if base_space is None:
        base_space = HYPERPARAM_SEARCH_SPACE

    refined = {}

    for name, spec in base_space.items():
        if name not in top_rows.columns:
            refined[name] = copy.deepcopy(spec)
            continue

        values = top_rows[name].dropna()
        if values.empty:
            refined[name] = copy.deepcopy(spec)
            continue

        if spec["type"] == "choice":
            original_values = list(spec["values"])
            observed_values = []

            for value in values.tolist():
                if isinstance(value, np.generic):
                    value = value.item()

                if original_values and isinstance(original_values[0], bool):
                    value = bool(value)
                elif original_values and isinstance(original_values[0], int) and not isinstance(original_values[0], bool):
                    value = int(value)
                elif original_values and isinstance(original_values[0], float):
                    value = float(value)

                if value in original_values and value not in observed_values:
                    observed_values.append(value)

            refined[name] = {
                "type": "choice",
                "values": observed_values if observed_values else original_values,
            }

        elif spec["type"] in {"uniform", "log_uniform"}:
            numeric_values = values.astype(float)
            observed_low = float(numeric_values.min())
            observed_high = float(numeric_values.max())

            base_low = float(spec["low"])
            base_high = float(spec["high"])

            if observed_low == observed_high:
                span = abs(observed_low) * expand_ratio
                if span == 0:
                    span = (base_high - base_low) * expand_ratio
            else:
                span = observed_high - observed_low

            new_low = observed_low - span * expand_ratio
            new_high = observed_high + span * expand_ratio

            new_low = max(base_low, new_low)
            new_high = min(base_high, new_high)

            if spec["type"] == "log_uniform":
                new_low = max(new_low, 1e-12)

            if new_low >= new_high:
                refined[name] = copy.deepcopy(spec)
            else:
                refined[name] = {
                    "type": spec["type"],
                    "low": float(new_low),
                    "high": float(new_high),
                }

        else:
            refined[name] = copy.deepcopy(spec)

    return refined


def dedupe_scored_configs(scored_df, key_col="hyperparams_json"):
    if scored_df.empty:
        return scored_df

    return (
        scored_df
        .sort_values("hyperband_score", ascending=False)
        .drop_duplicates(subset=[key_col], keep="first")
        .reset_index(drop=True)
    )


def sample_unique_configs(rng, experiment_name, count, search_space, seen, prefix, origin):
    configs = []
    attempts = 0
    max_attempts = max(100, count * 100)

    while len(configs) < count and attempts < max_attempts:
        attempts += 1

        hp = sample_hyperparams(rng, search_space=search_space)
        hp_json = hyperparams_to_json(hp)

        if hp_json in seen:
            continue

        seen.add(hp_json)
        configs.append({
            "config_id": f"{experiment_name}_{prefix}_{len(configs) + 1:03d}",
            "hyperparams": hp,
            "origin": origin,
        })

    return configs


def hyperparams_to_json(hp):
    return json.dumps(hp, sort_keys=True)


def compute_hyperband_scores(rows):
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df
    interval_mape_scaled = df["val_interval_mape"] / 100.0
    df["hyperband_score"] = df["val_scan_micro_auc"] - (interval_mape_scaled / (1.0 + interval_mape_scaled))
    df["score_formula"] = HYPERBAND_SCORE_NAME
    return df.sort_values("hyperband_score", ascending=False).reset_index(drop=True)

def hyperband_search(experiment, split_patients):
    rng = np.random.default_rng(HYPERBAND_RANDOM_SEED + sum(ord(ch) for ch in experiment["name"]))
    active_configs = []
    seen = set()
    while len(active_configs) < HYPERBAND_INITIAL_CONFIGS:
        hp = sample_hyperparams(rng)
        hp_json = hyperparams_to_json(hp)
        if hp_json in seen:
            continue
        seen.add(hp_json)
        active_configs.append({
            "config_id": f"{experiment['name']}_cfg_{len(active_configs) + 1:03d}",
            "hyperparams": hp,
            "origin": "initial",
        })

    all_rows = []
    rung_summaries = []
    subset_seed = HYPERBAND_RANDOM_SEED + 1000 + sum(ord(ch) for ch in experiment["name"])
    for rung_idx, train_fraction in enumerate(HYPERBAND_RUNG_TRAIN_FRACTIONS, start=1):
        stage_cfg = HYPERBAND_STAGE_SCHEDULE[rung_idx]
        stage_iterations = int(stage_cfg["iterations"])
        stage_search_space = copy.deepcopy(HYPERPARAM_SEARCH_SPACE)
        rung_raw_rows = []

        for stage_iter in range(1, stage_iterations + 1):
            print(
                f"[Data-Hyperband:{experiment['name']}] "
                f"rung={rung_idx} iter={stage_iter}/{stage_iterations} "
                f"configs={len(active_configs)} train_fraction={train_fraction:.2f} "
                f"epochs={HYPERBAND_EPOCHS_PER_TRIAL}"
            )

            iter_rows = []

            for cfg in active_configs:
                result = run_experiment(
                    experiment,
                    split_patients=split_patients,
                    hyperparams=cfg["hyperparams"],
                    train_fraction=train_fraction,
                    train_subset_seed=subset_seed,
                    max_epochs=HYPERBAND_EPOCHS_PER_TRIAL,
                    evaluate_test=False,
                    trial_id=f"{cfg['config_id']}_r{rung_idx}_i{stage_iter}",
                )
                metrics = result["val_metrics"]
                row = {
                    "experiment": experiment["name"],
                    "rung": rung_idx,
                    "stage_iteration": stage_iter,
                    "candidate_origin": cfg.get("origin", "unknown"),
                    "train_fraction": float(train_fraction),
                    "epochs_per_trial": HYPERBAND_EPOCHS_PER_TRIAL,
                    "train_patients_used": result["split_sizes"]["train_patients"],
                    "train_patients_full": result["split_sizes"]["train_patients_full"],
                    "train_examples_used": result["split_sizes"]["train_examples"],
                    "train_examples_full": result["split_sizes"]["train_examples_full"],
                    "config_id": cfg["config_id"],
                    "best_val_loss": result["best_val_loss"],
                    "val_interval_mape": metrics.get("interval_mape", np.nan),
                    "val_scan_micro_auc": metrics.get("scan_micro_auc", np.nan),
                    "val_interval_mae": metrics.get("interval_mae", np.nan),
                    "val_interval_rmse": metrics.get("interval_rmse", np.nan),
                    "val_acc@0.5m": metrics.get("acc@0.5m", np.nan),
                    "val_acc@1m": metrics.get("acc@1m", np.nan),
                    "val_acc@2m": metrics.get("acc@2m", np.nan),
                    "hyperparams_json": hyperparams_to_json(cfg["hyperparams"]),
                    **cfg["hyperparams"],
                }

                iter_rows.append(row)
                del result

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            scored = compute_hyperband_scores(iter_rows)
            iter_unique_scored = dedupe_scored_configs(scored)

            all_rows.extend(scored.to_dict("records"))
            rung_raw_rows.extend(scored.to_dict("records"))

            iter_path = OUTPUT_DIR / f"data_hyperband_trials_{experiment['name']}_rung_{rung_idx}_iter_{stage_iter}.csv"
            scored.to_csv(iter_path, index=False)

            append_report_section(
                f"hyperband::{experiment['name']}::rung_{rung_idx}::iter_{stage_iter}",
                iter_unique_scored.head(10).to_string(index=False),
            )

            if stage_iter < stage_iterations:
                refine_n = min(int(stage_cfg["refine_top_k"]), len(iter_unique_scored))
                elite_n = min(int(stage_cfg["elite_keep"]), len(iter_unique_scored))
                refresh_count = int(stage_cfg["refresh_configs"])

                refine_rows = iter_unique_scored.head(refine_n)
                elite_rows = iter_unique_scored.head(elite_n)

                stage_search_space = make_refined_search_space(
                    refine_rows,
                    base_space=HYPERPARAM_SEARCH_SPACE,
                    expand_ratio=HYPERBAND_REFINE_EXPAND_RATIO,
                )

                elites = [
                    {
                        "config_id": row["config_id"],
                        "hyperparams": json.loads(row["hyperparams_json"]),
                        "origin": "elite",
                    }
                    for _, row in elite_rows.iterrows()
                ]

                new_configs = sample_unique_configs(
                    rng,
                    experiment["name"],
                    refresh_count,
                    stage_search_space,
                    seen,
                    prefix=f"r{rung_idx}_i{stage_iter + 1}_new",
                    origin="refined_sample",
                )

                active_configs = elites + new_configs

        rung_scored_all = dedupe_scored_configs(pd.DataFrame(rung_raw_rows))

        keep_n = min(int(stage_cfg["next_stage_keep"]), len(rung_scored_all))
        keep_rows = rung_scored_all.head(keep_n)

        active_configs = [
            {
                "config_id": row["config_id"],
                "hyperparams": json.loads(row["hyperparams_json"]),
                "origin": "rung_survivor" if rung_idx < len(HYPERBAND_RUNG_TRAIN_FRACTIONS) else "final_best",
            }
            for _, row in keep_rows.iterrows()
        ]

        rung_path = OUTPUT_DIR / f"data_hyperband_trials_{experiment['name']}_rung_{rung_idx}.csv"
        rung_scored_all.to_csv(rung_path, index=False)

        rung_summaries.append({
            "experiment": experiment["name"],
            "rung": rung_idx,
            "stage_iterations": stage_iterations,
            "train_fraction": float(train_fraction),
            "epochs_per_trial": HYPERBAND_EPOCHS_PER_TRIAL,
            "train_patients_used": int(rung_scored_all.iloc[0]["train_patients_used"]),
            "train_patients_full": int(rung_scored_all.iloc[0]["train_patients_full"]),
            "train_examples_used": int(rung_scored_all.iloc[0]["train_examples_used"]),
            "train_examples_full": int(rung_scored_all.iloc[0]["train_examples_full"]),
            "evaluated_rows": len(rung_raw_rows),
            "unique_configs": len(rung_scored_all),
            "kept_configs": len(active_configs),
            "best_config_id": rung_scored_all.iloc[0]["config_id"],
            "best_hyperband_score": rung_scored_all.iloc[0]["hyperband_score"],
            "best_val_scan_micro_auc": rung_scored_all.iloc[0]["val_scan_micro_auc"],
            "best_val_interval_mae": rung_scored_all.iloc[0]["val_interval_mae"],
        })

    all_df = pd.DataFrame(all_rows)
    final_rung = len(HYPERBAND_RUNG_TRAIN_FRACTIONS)
    final_scored = dedupe_scored_configs(
        all_df[all_df["rung"] == final_rung],
        key_col="hyperparams_json",
    )
    best_row = final_scored.iloc[0].to_dict()
    best_hp = json.loads(best_row["hyperparams_json"])
    return best_hp, best_row, pd.DataFrame(all_rows), pd.DataFrame(rung_summaries)


experiment_results = []
summary_rows = []
metrics_rows = []
hyperband_trial_frames = []
hyperband_best_rows = []
hyperband_final_best_rows = []
hyperband_rung_frames = []
print(f"[Runner] Starting data-Hyperband for {len(EXPERIMENTS)} experiments...")
for exp_idx, experiment in enumerate(EXPERIMENTS, start=1):
    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) tuning {experiment['name']}")
    best_hp, best_row, trials_df, rung_df = hyperband_search(experiment, shared_split_patients)
    hyperband_trial_frames.append(trials_df)
    hyperband_rung_frames.append(rung_df)
    hyperband_best_rows.append(best_row)

    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) final training with best hyperparameters for {experiment['name']}")
    result = run_experiment(
        experiment,
        split_patients=shared_split_patients,
        hyperparams=best_hp,
        train_fraction=1.0,
        train_subset_seed=HYPERBAND_RANDOM_SEED,
        max_epochs=EPOCHS,
        evaluate_test=True,
        trial_id="best_final",
    )
    result["scan_thresholds"] = tune_scan_thresholds(
        result["val_pred"]["y_scan"], result["val_pred"]["scan_probs"]
    )
    result["val_metrics"].update(compute_scan_f1(result["val_pred"], result["scan_thresholds"]))
    result["test_metrics"].update(compute_scan_f1(result["pred"], result["scan_thresholds"]))
    experiment_results.append(result)
    final_best_record = dict(best_row)
    final_best_record.update({f"test_{k}": v for k, v in result["test_metrics"].items()})
    final_best_record["final_train_epochs"] = EPOCHS
    final_best_record["final_best_val_loss"] = result["best_val_loss"]
    hyperband_final_best_rows.append(final_best_record)
    summary_rows.append({
        "experiment": result["name"],
        "feature_tables": ", ".join(result["feature_tables"]),
        "train_examples": result["split_sizes"]["train_examples"],
        "val_examples": result["split_sizes"]["val_examples"],
        "test_examples": result["split_sizes"]["test_examples"],
        "best_hyperband_config_id": best_row["config_id"],
        "best_hyperband_score": best_row["hyperband_score"],
    })
    metrics_rows.append({"experiment": result["name"], **result["test_metrics"]})
    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) finished {experiment['name']}")

hyperband_trials_df = pd.concat(hyperband_trial_frames, ignore_index=True) if hyperband_trial_frames else pd.DataFrame()
hyperband_best_df = pd.DataFrame(hyperband_best_rows)
hyperband_final_best_df = pd.DataFrame(hyperband_final_best_rows)
hyperband_rungs_df = pd.concat(hyperband_rung_frames, ignore_index=True) if hyperband_rung_frames else pd.DataFrame()
save_dataframe_artifact(hyperband_trials_df, "hyperband_trials", title="hyperband_trials", index=False)
save_dataframe_artifact(hyperband_best_df, "hyperband_best_by_experiment", title="hyperband_best_by_experiment", index=False)
save_dataframe_artifact(hyperband_final_best_df, "hyperband_final_best_by_experiment", title="hyperband_final_best_by_experiment", index=False)
save_dataframe_artifact(hyperband_rungs_df, "hyperband_rung_summary", title="hyperband_rung_summary", index=False)

prediction_artifact_rows = []
print("[Runner] Saving per-experiment artifacts...")
for result in experiment_results:
    print(f"[Runner] saving artifacts for {result['name']}")
    save_dataframe_artifact(result["history"], f"training_history_{result['name']}", title=f"training_history::{result['name']}", index=False)

    scan_thresholds_df = pd.DataFrame({
        "scan_type": SCAN_TYPE_NAMES,
        "threshold": result["scan_thresholds"],
    })
    save_dataframe_artifact(scan_thresholds_df, f"validation_scan_thresholds_{result['name']}",
                            title=f"validation_scan_thresholds::{result['name']}", index=False)
    validation_detail_df = build_prediction_detail_df(
        result["val_pred"], result["interval_thresholds"], result["scan_thresholds"]
    )
    save_dataframe_artifact(validation_detail_df, f"validation_prediction_detail_{result['name']}",
                            title=f"validation_prediction_detail::{result['name']}", index=False)
    detail_df = build_prediction_detail_df(
        result["pred"], result["interval_thresholds"], result["scan_thresholds"]
    )
    detail_stem = f"prediction_detail_{result['name']}"
    detail_csv_path, detail_txt_path = save_dataframe_artifact(
        detail_df,
        detail_stem,
        title=f"prediction_detail::{result['name']}",
        index=False,
    )
    incorrect_df = build_incorrect_prediction_df(
        detail_df,
        result["test_examples_df"],
        result["name"],
        result["feature_tables"],
        result["input_flags"],
        result["interval_thresholds"],
    )
    incorrect_stem = f"incorrect_predictions_{result['name']}"
    incorrect_csv_path, incorrect_txt_path = save_dataframe_artifact(
        incorrect_df,
        incorrect_stem,
        title=f"incorrect_predictions::{result['name']}",
        index=False,
    )

    category_df, _ = compute_category_contributions(result)
    exact_df, _ = compute_exact_feature_contributions(result)

    category_stem = f"category_contribution_{result['name']}"
    category_csv_path, category_txt_path = save_dataframe_artifact(
        category_df,
        category_stem,
        title=f"category_contribution::{result['name']}",
        index=False,
    )

    exact_stem = f"exact_feature_contribution_{result['name']}"
    exact_csv_path, exact_txt_path = save_dataframe_artifact(
        exact_df,
        exact_stem,
        title=f"exact_feature_contribution::{result['name']}",
        index=False,
    )

    metrics_df = pd.DataFrame([{ "experiment": result["name"], **result["test_metrics"] }])
    metrics_stem = f"experiment_metrics_{result['name']}"
    metrics_csv_path, metrics_txt_path = save_dataframe_artifact(
        metrics_df,
        metrics_stem,
        title=f"experiment_metrics::{result['name']}",
        index=False,
    )

    prediction_artifact_rows.append(
        {
            "experiment": result["name"],
            "feature_tables": ", ".join(result["feature_tables"]),
            "prediction_detail_csv": str(detail_csv_path),
            "prediction_detail_txt": str(detail_txt_path),
            "incorrect_prediction_csv": str(incorrect_csv_path),
            "incorrect_prediction_txt": str(incorrect_txt_path),
            "category_contribution_csv": str(category_csv_path),
            "category_contribution_txt": str(category_txt_path),
            "exact_feature_contribution_csv": str(exact_csv_path),
            "exact_feature_contribution_txt": str(exact_txt_path),
            "metrics_csv": str(metrics_csv_path),
            "metrics_txt": str(metrics_txt_path),
            "n_rows": len(detail_df),
            "n_incorrect": len(incorrect_df),
            "n_category_rows": len(category_df),
            "n_exact_rows": len(exact_df),
        }
    )

summary_df = pd.DataFrame(summary_rows)
metrics_summary_df = pd.DataFrame(metrics_rows)
prediction_artifacts_df = pd.DataFrame(prediction_artifact_rows)
save_dataframe_artifact(summary_df, "experiment_summary", title="experiment_summary", index=False)
save_dataframe_artifact(metrics_summary_df, "experiment_metrics_summary", title="experiment_metrics_summary", index=False)
save_dataframe_artifact(prediction_artifacts_df, "prediction_artifacts", title="prediction_artifacts", index=False)

plot_metadata = {
    "scan_type_names": SCAN_TYPE_NAMES,
    "interval_metric_names": ["MAE", "MAPE"],
    "interval_acc_thresholds": INTERVAL_ACC_THRESHOLDS,
    "scan_threshold_selection": "Per-scan validation F1 on 0.05–0.95 in steps of 0.05; 0.5 fallback for a single-class validation target",
    "hyperband_score": HYPERBAND_SCORE_NAME,
    "hyperband_score_formula": "val_scan_micro_auc - ((val_interval_mape / 100.0) / (1.0 + (val_interval_mape / 100.0)))",
    "hyperband_resource": "patient_wise_train_fraction",
    "hyperband_rung_train_fractions": HYPERBAND_RUNG_TRAIN_FRACTIONS,
    "hyperband_epochs_per_trial": HYPERBAND_EPOCHS_PER_TRIAL,
    "hyperband_initial_configs": HYPERBAND_INITIAL_CONFIGS,
    "hyperband_stage_schedule": HYPERBAND_STAGE_SCHEDULE,
    "hyperband_refine_expand_ratio": HYPERBAND_REFINE_EXPAND_RATIO,
    "hyperband_unique_config_key": "hyperparams_json",
    "hyperband_selection_scope": "rank unique configs across all iterations within each rung",
}
with open(OUTPUT_DIR / "plot_metadata.json", "w", encoding="utf-8") as f:
    json.dump(plot_metadata, f, ensure_ascii=False, indent=2)
append_report_section("saved_plot_metadata", str(OUTPUT_DIR / "plot_metadata.json"))

display(summary_df)
display(metrics_summary_df)
display(hyperband_best_df)
display(hyperband_final_best_df)
display(prediction_artifacts_df)





