#!/usr/bin/env python
# coding: utf-8

"""
Tabular next-scan scan-type plus interval prediction with data-Hyperband.

This runner mirrors the GRU C+R normalized-score experiment structure, but reads
the flattened tabular multitask tables and predicts scan type plus interval.

Example:
  EXPERIMENT_NAME=DSIMB MULTITASK_TABULAR_DATA_DIR=/restricted/tabular python linear_multitask.py
"""

import os
import io
import json
import math
import random
import re
import tempfile
import warnings
import contextlib
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_ml4h"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

with contextlib.redirect_stderr(io.StringIO()):
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import auc, f1_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    env_data_dir = os.environ.get("MULTITASK_TABULAR_DATA_DIR") or os.environ.get("MULTITASK_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir).expanduser().resolve())
    candidates = [
        *candidates,
        SCRIPT_DIR / "data" / "multitask_tabular",
        SCRIPT_DIR / "data" / "prepared" / "multitask_tabular",
        CODE_DIR / "data" / "prepared" / "multitask_tabular",
        cwd / "data" / "prepared" / "multitask_tabular",
        cwd / "data" / "multitask_tabular",
        cwd / "Code" / "hopper_cluster" / "data" / "multitask_tabular",
        cwd / "Code" / "data" / "prepared" / "multitask_tabular",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find prepared multitask_tabular data directory from cwd={cwd}")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = resolve_data_dir()
MODEL_NAME = "lr"
TABLE_PATHS = {
    "target": DATA_DIR / "target.csv",
    "demographic": DATA_DIR / "demographic.csv",
    "imaging_history": DATA_DIR / "imaging_history.csv",
    "staging_history": DATA_DIR / "staging_history.csv",
    "mastectomy_history": DATA_DIR / "mastectomy_history.csv",
    "biomarker_history": DATA_DIR / "biomarker_history.csv",
}

for table_name, path in TABLE_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared tabular CSV table {table_name}: {path}")

SELECTED_EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "").strip()
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_EXPERIMENT_TAG = re.sub(r"[^A-Za-z0-9_.-]+", "_", SELECTED_EXPERIMENT_NAME or "ALL")
RESULTS_ROOT = Path(os.environ.get("MODEL_RESULTS_DIR", SCRIPT_DIR.parents[1] / "restricted_artifacts" / "results")).expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / f"{MODEL_NAME}_tabular_multitask_compare_next_scan_data_hyperband_normalized_score_{RUN_TIMESTAMP}_{RUN_EXPERIMENT_TAG}_pid{os.getpid()}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "report.txt"
REPORT_PATH.write_text("", encoding="utf-8")

SCAN_TYPE_NAMES = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
N_SCAN_TYPES = len(SCAN_TYPE_NAMES)
SCAN_THRESHOLD = 0.5
INTERVAL_ACC_THRESHOLDS = [0.5, 1.0, 2.0]
HYPERBAND_SCORE_NAME = "scan_auc_minus_normalized_interval_mape"

EXPERIMENTS = [
    {"name": "DS", "feature_tables": ["demographic", "staging_history"]},
    {"name": "DSI", "feature_tables": ["demographic", "staging_history", "imaging_history"]},
    {"name": "DSM", "feature_tables": ["demographic", "staging_history", "mastectomy_history"]},
    {"name": "DSB", "feature_tables": ["demographic", "staging_history", "biomarker_history"]},
    {"name": "DSIM", "feature_tables": ["demographic", "staging_history", "imaging_history", "mastectomy_history"]},
    {"name": "DSIB", "feature_tables": ["demographic", "staging_history", "imaging_history", "biomarker_history"]},
    {"name": "DSMB", "feature_tables": ["demographic", "staging_history", "mastectomy_history", "biomarker_history"]},
    {"name": "DSIMB", "feature_tables": ["demographic", "staging_history", "imaging_history", "mastectomy_history", "biomarker_history"]},
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

HYPERBAND_ENABLED = True
HYPERBAND_RUNG_TRAIN_FRACTIONS = [0.25, 0.50, 1.00]
HYPERBAND_INITIAL_CONFIGS = int(os.environ.get("HYPERBAND_INITIAL_CONFIGS", "36"))
HYPERBAND_RANDOM_SEED = SEED
HYPERBAND_REFINE_EXPAND_RATIO = 0.20

HYPERBAND_STAGE_SCHEDULE = {
    1: {"iterations": 4, "next_stage_keep": 12, "refine_top_k": 12, "elite_keep": 2, "refresh_configs": 10},
    2: {"iterations": 3, "next_stage_keep": 4, "refine_top_k": 4, "elite_keep": 1, "refresh_configs": 3},
    3: {"iterations": 3, "next_stage_keep": 1, "refine_top_k": 2, "elite_keep": 1, "refresh_configs": 1},
}

MLP_FINAL_EPOCHS = int(os.environ.get("MLP_FINAL_EPOCHS", "20"))
MLP_TRIAL_EPOCHS = int(os.environ.get("MLP_TRIAL_EPOCHS", str(MLP_FINAL_EPOCHS)))
MLP_PATIENCE = int(os.environ.get("MLP_PATIENCE", "5"))

LR_DEFAULT_HYPERPARAMS = {
    "scan_C": 1.0,
    "scan_penalty": "l2",
    "scan_solver": "lbfgs",
    "scan_l1_ratio": 0.5,
    "class_weight": None,
    "numeric_imputer": "constant",
    "numeric_imputer_fill_value": -1.0,
    "categorical_imputer": "constant",
    "categorical_imputer_fill_value": "__MISSING__",
    "scale_numeric": True,
}

RIDGE_DEFAULT_HYPERPARAMS = {
    "interval_model": "ridge",
    "interval_alpha": 1.0,
}

DEFAULT_HYPERPARAMS = {**LR_DEFAULT_HYPERPARAMS, **RIDGE_DEFAULT_HYPERPARAMS}

LR_HYPERPARAM_SEARCH_SPACE = {
    "scan_C": {
        "type": "choice",
        "values": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    },
    "class_weight": {"type": "choice", "values": [None, "balanced"]},
}

RIDGE_HYPERPARAM_SEARCH_SPACE = {
    "interval_alpha": {
        "type": "choice",
        "values": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    },
}

HYPERPARAM_SEARCH_SPACE = {**LR_HYPERPARAM_SEARCH_SPACE, **RIDGE_HYPERPARAM_SEARCH_SPACE}


def append_report_section(title, content):
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(str(title) + "\n")
        f.write("=" * 80 + "\n")
        f.write(str(content) + "\n")


def save_dataframe_artifact(df, stem, title=None, index=False):
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    txt_path = OUTPUT_DIR / f"{stem}.txt"
    df.to_csv(csv_path, index=index)
    txt_path.write_text(df.to_string(index=index), encoding="utf-8")
    if title:
        append_report_section(title, df.to_string(index=index))
    return csv_path, txt_path


def parse_scan_vector(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = value
    else:
        text = str(value).strip().strip("[]")
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    arr = np.zeros(N_SCAN_TYPES, dtype=np.float32)
    for idx, part in enumerate(parts[:N_SCAN_TYPES]):
        try:
            arr[idx] = float(part)
        except Exception:
            arr[idx] = 0.0
    return arr


def safe_multilabel_micro_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or len(np.unique(y_true.ravel())) < 2:
        return None, None, np.nan
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    return fpr, tpr, auc(fpr, tpr)


def safe_multilabel_macro_curve(y_true, y_prob):
    aucs = []
    for idx in range(y_true.shape[1]):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_prob[:, idx])
        aucs.append(auc(fpr, tpr))
    return np.nan if not aucs else float(np.mean(aucs))


def tune_scan_thresholds(y_true, y_prob):
    thresholds = []
    for class_idx in range(y_true.shape[1]):
        best_threshold = SCAN_THRESHOLD
        best_f1 = -1.0
        if len(np.unique(y_true[:, class_idx])) < 2:
            thresholds.append(best_threshold)
            continue
        for threshold in np.linspace(0.05, 0.95, 19):
            pred = (y_prob[:, class_idx] >= threshold).astype(int)
            score = f1_score(y_true[:, class_idx], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype=np.float32)


def compute_interval_metrics(pred, thresholds):
    y_true = np.asarray(pred["interval_true"], dtype=float)
    y_hat = np.asarray(pred["interval_pred"], dtype=float)
    abs_error = np.abs(y_true - y_hat)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2))) if len(y_true) else np.nan
    denom = np.maximum(np.abs(y_true), 1e-6)
    metrics = {
        "interval_mae": float(np.mean(abs_error)) if len(abs_error) else np.nan,
        "interval_rmse": rmse,
        "interval_mape": float(np.mean(abs_error / denom) * 100.0) if len(abs_error) else np.nan,
    }
    for threshold in thresholds:
        key = f"acc@{float(threshold):g}m"
        metrics[key] = float((abs_error <= float(threshold)).mean()) if len(abs_error) else np.nan
    return metrics


def compute_task_metrics(pred, thresholds):
    y_true = np.asarray(pred["scan_true"], dtype=int)
    y_prob = np.asarray(pred["scan_prob"], dtype=float)
    scan_pred = (y_prob >= np.asarray(pred.get("scan_thresholds", np.repeat(SCAN_THRESHOLD, N_SCAN_TYPES)))).astype(int)
    _, _, micro_auc = safe_multilabel_micro_curve(y_true, y_prob)
    macro_auc = safe_multilabel_macro_curve(y_true, y_prob)
    return {
        "scan_micro_auc": float(micro_auc) if not pd.isna(micro_auc) else np.nan,
        "scan_macro_auc": float(macro_auc) if not pd.isna(macro_auc) else np.nan,
        "scan_micro_f1": float(f1_score(y_true.ravel(), scan_pred.ravel(), zero_division=0)),
        "scan_macro_f1": float(f1_score(y_true, scan_pred, average="macro", zero_division=0)),
        **compute_interval_metrics(pred, thresholds),
    }


def read_prepared_table(name, path):
    print(f"[Data] Using CSV table {path.name} for {name}.")
    return pd.read_csv(path)


def load_tables():
    return {name: read_prepared_table(name, path) for name, path in TABLE_PATHS.items()}


def merge_feature_table(base_df, feature_df, table_name):
    if table_name == "demographic":
        return base_df.merge(feature_df, on="patient_id", how="left", suffixes=("", f"_{table_name}"))
    key_cols = ["patient_id", "anchor_scan_date", "anchor_sequence", "target_sequence"]
    return base_df.merge(feature_df, on=key_cols, how="left", suffixes=("", f"_{table_name}"))


def build_examples(tables, experiment):
    target = tables["target"].copy()
    examples = target.copy()
    for table_name in experiment["feature_tables"]:
        examples = merge_feature_table(examples, tables[table_name], table_name)
    examples = expand_scan_vector_feature_columns(examples)
    examples["target_scan_array"] = examples["target_scan_vector"].apply(parse_scan_vector)
    if "target_interval_months" not in examples.columns:
        examples["target_interval_months"] = examples["target_gap_months"]
    return examples


def expand_scan_vector_feature_columns(df):
    out = df.copy()
    scan_vector_cols = [
        col for col in out.columns
        if col.endswith("_scan_vector") and col != "target_scan_vector"
    ]
    for col in scan_vector_cols:
        parsed = out[col].apply(parse_scan_vector)
        values = np.vstack(parsed.to_numpy()).astype(np.float32)
        base = col[: -len("_scan_vector")]
        for idx, scan_name in enumerate(SCAN_TYPE_NAMES):
            out[f"{base}_{scan_name}"] = values[:, idx]
    return out.drop(columns=scan_vector_cols)


def make_patient_splits(patient_ids):
    patient_ids = sorted(pd.Series(patient_ids).dropna().astype(str).unique())
    train_patients, temp_patients = train_test_split(patient_ids, test_size=0.30, random_state=SEED)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.50, random_state=SEED)
    return {"train": set(train_patients), "val": set(val_patients), "test": set(test_patients)}


def split_examples(examples_df, split_patients):
    train_examples = examples_df[examples_df["patient_id"].astype(str).isin(split_patients["train"])].reset_index(drop=True)
    val_examples = examples_df[examples_df["patient_id"].astype(str).isin(split_patients["val"])].reset_index(drop=True)
    test_examples = examples_df[examples_df["patient_id"].astype(str).isin(split_patients["test"])].reset_index(drop=True)
    if len(train_examples) == 0 or len(val_examples) == 0 or len(test_examples) == 0:
        raise ValueError(f"Empty split detected: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}")
    return train_examples, val_examples, test_examples


def subsample_train_examples_by_patient(train_examples, fraction=1.0, seed=SEED):
    if fraction >= 1.0:
        return train_examples.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    patient_ids = np.array(sorted(train_examples["patient_id"].astype(str).unique()))
    keep_n = max(1, int(math.ceil(len(patient_ids) * float(fraction))))
    keep_patients = set(rng.choice(patient_ids, size=keep_n, replace=False))
    return train_examples[train_examples["patient_id"].astype(str).isin(keep_patients)].reset_index(drop=True)


def get_feature_columns(df):
    drop_cols = {
        "patient_id",
        "anchor_scan_date",
        "anchor_sequence",
        "target_sequence",
        "target_scan_type_type",
        "target_scan_vector",
        "target_scan_array",
        "target_gap_months",
        "target_interval_months",
        "target_interval_bin",
    }
    return [col for col in df.columns if col not in drop_cols and not col.endswith("_date")]


def normalize_object_value(value):
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text == "" or text.lower() in {"nan", "nat", "none", "<na>"} else text


def split_feature_types(train_df, feature_cols):
    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        series = train_df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        if non_null > 0 and converted.notna().sum() / non_null >= 0.95:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def prepare_feature_frames(train_df, val_df, test_df, feature_cols):
    frames = [train_df[feature_cols].copy(), val_df[feature_cols].copy(), test_df[feature_cols].copy()]
    numeric_cols, categorical_cols = split_feature_types(train_df, feature_cols)
    for frame in frames:
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in categorical_cols:
            frame[col] = frame[col].map(normalize_object_value)
    return frames[0], frames[1], frames[2], numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols, hp):
    num_steps = [(
        "imputer",
        SimpleImputer(
            strategy=hp.get("numeric_imputer", "constant"),
            fill_value=float(hp.get("numeric_imputer_fill_value", -1.0)),
        ),
    )]
    if hp.get("scale_numeric", False):
        num_steps.append(("scaler", StandardScaler()))
    cat_imputer_strategy = hp.get("categorical_imputer", "constant")
    cat_fill = str(hp.get("categorical_imputer_fill_value", "__MISSING__")) if cat_imputer_strategy == "constant" else None
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy=cat_imputer_strategy, fill_value=cat_fill)),
                ("onehot", encoder),
            ]),
            categorical_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def targets_from_examples(df):
    scan_y = np.vstack(df["target_scan_array"].to_numpy()).astype(np.float32)
    interval_y = df["target_interval_months"].astype(float).to_numpy(dtype=np.float32)
    return scan_y, interval_y


def make_ridge_interval_model(hp):
    return Ridge(alpha=float(hp.get("interval_alpha", 1.0)), random_state=SEED)


class SafeBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1], dtype=int)
        observed = np.unique(y)
        if len(observed) < 2:
            self.constant_class_ = int(observed[0]) if len(observed) else 0
            self.estimator_ = None
        else:
            self.constant_class_ = None
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        if self.estimator_ is None:
            prob_one = float(self.constant_class_ == 1)
            return np.column_stack([np.full(n, 1.0 - prob_one), np.full(n, prob_one)])
        prob = self.estimator_.predict_proba(X)
        if prob.shape[1] == 2:
            return prob
        if len(getattr(self.estimator_, "classes_", [])) == 1 and int(self.estimator_.classes_[0]) == 1:
            return np.column_stack([np.zeros(n), np.ones(n)])
        return np.column_stack([np.ones(n), np.zeros(n)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_lr_scan_model(hp):
    penalty = hp.get("scan_penalty", "l2")
    solver = hp.get("scan_solver", "lbfgs")
    # Enforce valid solver/penalty combinations before model construction.
    if solver == "lbfgs":
        penalty = "l2"
    elif solver == "liblinear" and penalty == "elasticnet":
        penalty = "l2"
    elif penalty == "elasticnet" and solver != "saga":
        solver = "saga"
    l1_ratio = float(hp.get("scan_l1_ratio", 0.5)) if penalty == "elasticnet" else None
    base = LogisticRegression(
        C=float(hp.get("scan_C", 1.0)),
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        class_weight=hp.get("class_weight", None),
        max_iter=3000,
        random_state=SEED,
        n_jobs=1,
    )
    return MultiOutputClassifier(SafeBinaryClassifier(base), n_jobs=1)


def sklearn_scan_predict_proba(model, X):
    prob_list = model.predict_proba(X)
    cols = []
    for item in prob_list:
        arr = np.asarray(item)
        if arr.ndim == 1:
            cols.append(arr)
        elif arr.shape[1] == 1:
            cols.append(arr[:, 0])
        else:
            cols.append(arr[:, 1])
    return np.vstack(cols).T.astype(np.float32)


def run_experiment(experiment, tables, split_patients, hyperparams=None, train_fraction=1.0, train_subset_seed=SEED, max_epochs=None, evaluate_test=True, trial_id=None):
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    run_label = f"{MODEL_NAME}:{experiment['name']}" if trial_id is None else f"{MODEL_NAME}:{experiment['name']}:{trial_id}"
    print(f"[{run_label}] building tabular examples")
    examples_df = build_examples(tables, experiment)
    train_full, val_examples, test_examples = split_examples(examples_df, split_patients)
    train_examples = subsample_train_examples_by_patient(train_full, fraction=train_fraction, seed=train_subset_seed)
    train_patients_full = train_full["patient_id"].nunique()
    train_patients = train_examples["patient_id"].nunique()
    print(
        f"[{run_label}] train_examples={len(train_examples)}/{len(train_full)} "
        f"train_patients={train_patients}/{train_patients_full} train_fraction={float(train_fraction):.2f} "
        f"val={len(val_examples)} test={len(test_examples)}"
    )

    feature_cols = get_feature_columns(train_examples)
    X_train_df, X_val_df, X_test_df, numeric_cols, categorical_cols = prepare_feature_frames(
        train_examples, val_examples, test_examples, feature_cols
    )
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, hp)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    y_scan_train, y_interval_train = targets_from_examples(train_examples)
    y_scan_val, y_interval_val = targets_from_examples(val_examples)
    y_scan_test, y_interval_test = targets_from_examples(test_examples)

    scan_model = make_lr_scan_model(hp)
    interval_model = make_ridge_interval_model(hp)
    scan_model.fit(X_train, y_scan_train.astype(int))
    interval_model.fit(X_train, y_interval_train)
    model = {"scan": scan_model, "interval": interval_model}
    history_df = pd.DataFrame([{"epoch": 1, "train_loss": np.nan, "val_loss": np.nan}])
    best_val_loss = np.nan
    val_scan_prob = sklearn_scan_predict_proba(scan_model, X_val)
    val_interval_pred = interval_model.predict(X_val)
    test_scan_prob = sklearn_scan_predict_proba(scan_model, X_test)
    test_interval_pred = interval_model.predict(X_test)

    scan_thresholds = tune_scan_thresholds(y_scan_val, val_scan_prob)
    interval_thresholds = np.asarray(INTERVAL_ACC_THRESHOLDS, dtype=float)
    val_pred = {
        "scan_true": y_scan_val,
        "scan_prob": val_scan_prob,
        "scan_thresholds": scan_thresholds,
        "interval_true": y_interval_val,
        "interval_pred": val_interval_pred,
    }
    val_metrics = compute_task_metrics(val_pred, interval_thresholds)
    print(f"[{run_label}] val metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items() if isinstance(v, (int, float, np.floating)) and not pd.isna(v)]))

    test_metrics = None
    test_pred = None
    if evaluate_test:
        test_pred = {
            "scan_true": y_scan_test,
            "scan_prob": test_scan_prob,
            "scan_thresholds": scan_thresholds,
            "interval_true": y_interval_test,
            "interval_pred": test_interval_pred,
        }
        test_metrics = compute_task_metrics(test_pred, interval_thresholds)
        print(f"[{run_label}] test metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating)) and not pd.isna(v)]))

    return {
        "name": experiment["name"],
        "model_name": MODEL_NAME,
        "feature_tables": experiment["feature_tables"],
        "input_flags": {name: name in experiment["feature_tables"] for name in TABLE_PATHS},
        "hyperparams": hp,
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "history": history_df,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pred": test_pred,
        "test_examples_df": test_examples,
        "interval_thresholds": interval_thresholds,
        "best_val_loss": best_val_loss,
        "split_sizes": {
            "train_examples": len(train_examples),
            "train_examples_full": len(train_full),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "train_patients": train_patients,
            "train_patients_full": train_patients_full,
            "train_fraction": float(train_fraction),
        },
    }


def build_prediction_detail_df(pred, interval_thresholds):
    y_true = np.asarray(pred["scan_true"], dtype=int)
    y_prob = np.asarray(pred["scan_prob"], dtype=float)
    thresholds = np.asarray(pred["scan_thresholds"], dtype=float)
    y_hat = (y_prob >= thresholds).astype(int)
    rows = []
    for idx in range(y_true.shape[0]):
        interval_abs_error = float(abs(pred["interval_true"][idx] - pred["interval_pred"][idx]))
        row = {
            "row_idx": idx,
            "true_scan_vector": y_true[idx].tolist(),
            "pred_scan_vector": y_hat[idx].tolist(),
            "interval_true_months": float(pred["interval_true"][idx]),
            "interval_pred_months": float(pred["interval_pred"][idx]),
            "interval_abs_error_months": interval_abs_error,
        }
        for scan_idx, scan_name in enumerate(SCAN_TYPE_NAMES):
            row[f"true_{scan_name}"] = int(y_true[idx, scan_idx])
            row[f"prob_{scan_name}"] = float(y_prob[idx, scan_idx])
            row[f"pred_{scan_name}"] = int(y_hat[idx, scan_idx])
            row[f"threshold_{scan_name}"] = float(thresholds[scan_idx])
        for threshold in interval_thresholds:
            row[f"interval_within_{float(threshold):g}m"] = bool(interval_abs_error <= float(threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def build_incorrect_prediction_df(detail_df, examples_df, experiment_name):
    df = detail_df.copy()
    example_cols = ["patient_id", "anchor_scan_date", "anchor_sequence", "target_sequence", "target_scan_vector", "target_interval_months"]
    examples_small = examples_df[example_cols].reset_index(drop=True)
    out = pd.concat([examples_small, df], axis=1)
    scan_wrong = out["true_scan_vector"].astype(str) != out["pred_scan_vector"].astype(str)
    interval_wrong = out["interval_abs_error_months"] > 1.0
    out = out[scan_wrong | interval_wrong].copy()
    out.insert(0, "experiment", experiment_name)
    return out


def sample_from_space(rng, spec):
    spec_type = spec["type"]
    if spec_type == "choice":
        values = spec["values"]
        return values[int(rng.integers(0, len(values)))]
    if spec_type == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    if spec_type == "log_uniform":
        low = math.log10(float(spec["low"]))
        high = math.log10(float(spec["high"]))
        return float(10 ** rng.uniform(low, high))
    raise ValueError(f"Unknown search space type: {spec_type}")


def sample_hyperparams(rng, search_space=None):
    space = search_space or HYPERPARAM_SEARCH_SPACE
    hp = dict(DEFAULT_HYPERPARAMS)
    for name, spec in space.items():
        hp[name] = sample_from_space(rng, spec)
    if hp.get("scan_solver") == "lbfgs":
        hp["scan_penalty"] = "l2"
    elif hp.get("scan_solver") == "liblinear" and hp.get("scan_penalty") == "elasticnet":
        hp["scan_penalty"] = "l2"
    elif hp.get("scan_penalty") == "elasticnet" and hp.get("scan_solver") != "saga":
        hp["scan_solver"] = "saga"
    return hp


def hyperparams_to_json(hp):
    return json.dumps(hp, sort_keys=True, default=str)


def make_refined_search_space(top_rows, base_space=None, expand_ratio=0.20):
    base = base_space or HYPERPARAM_SEARCH_SPACE
    if top_rows is None or len(top_rows) == 0:
        return base
    decoded = [json.loads(row["hyperparams_json"]) for _, row in top_rows.iterrows()]
    refined = {}
    for name, spec in base.items():
        if spec["type"] in {"choice"}:
            observed = [hp.get(name) for hp in decoded if name in hp]
            values = []
            for value in observed:
                if value not in values:
                    values.append(value)
            refined[name] = {"type": "choice", "values": values or spec["values"]}
            continue
        observed = np.asarray([float(hp.get(name)) for hp in decoded if name in hp and hp.get(name) is not None], dtype=float)
        if len(observed) == 0:
            refined[name] = spec
            continue
        lo = float(np.nanmin(observed))
        hi = float(np.nanmax(observed))
        span = max(hi - lo, 1e-12)
        lo = lo - span * expand_ratio
        hi = hi + span * expand_ratio
        if spec["type"] == "log_uniform":
            lo = max(lo, float(spec["low"]))
            hi = min(hi, float(spec["high"]))
            if lo <= 0 or hi <= lo:
                refined[name] = spec
            else:
                refined[name] = {"type": "log_uniform", "low": lo, "high": hi}
        else:
            lo = max(lo, float(spec["low"]))
            hi = min(hi, float(spec["high"]))
            refined[name] = {"type": "uniform", "low": lo, "high": hi} if hi > lo else spec
    return refined


def compute_hyperband_scores(rows):
    df = pd.DataFrame(rows).copy()
    interval_mape_scaled = df["val_interval_mape"].astype(float) / 100.0
    df["hyperband_score"] = df["val_scan_micro_auc"].astype(float) - (interval_mape_scaled / (1.0 + interval_mape_scaled))
    df["score_formula"] = HYPERBAND_SCORE_NAME
    return df.sort_values("hyperband_score", ascending=False).reset_index(drop=True)


def dedupe_scored_configs(scored_df, key_col="hyperparams_json"):
    if scored_df.empty:
        return scored_df
    return (
        scored_df.sort_values("hyperband_score", ascending=False)
        .drop_duplicates(subset=[key_col], keep="first")
        .reset_index(drop=True)
    )


def sample_unique_configs(rng, experiment_name, count, search_space, seen, prefix, origin):
    configs = []
    attempts = 0
    while len(configs) < int(count) and attempts < int(count) * 100:
        attempts += 1
        hp = sample_hyperparams(rng, search_space=search_space)
        key = hyperparams_to_json(hp)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "config_id": f"{experiment_name}_{prefix}_{len(configs) + 1:03d}",
            "hyperparams": hp,
            "origin": origin,
        })
    return configs


def hyperband_search(experiment, tables, split_patients):
    rng = np.random.default_rng(HYPERBAND_RANDOM_SEED + sum(ord(ch) for ch in experiment["name"]) + sum(ord(ch) for ch in MODEL_NAME))
    active_configs = []
    seen = set()
    initial_attempts = 0
    max_initial_attempts = max(int(HYPERBAND_INITIAL_CONFIGS) * 100, 100)
    while len(active_configs) < HYPERBAND_INITIAL_CONFIGS and initial_attempts < max_initial_attempts:
        initial_attempts += 1
        hp = sample_hyperparams(rng)
        key = hyperparams_to_json(hp)
        if key in seen:
            continue
        seen.add(key)
        active_configs.append({
            "config_id": f"{experiment['name']}_initial_{len(active_configs) + 1:03d}",
            "hyperparams": hp,
            "origin": "initial",
        })
    if not active_configs:
        raise ValueError("No unique Hyperband configs could be sampled from HYPERPARAM_SEARCH_SPACE.")

    all_rows = []
    rung_summaries = []
    subset_seed = HYPERBAND_RANDOM_SEED + 1000 + sum(ord(ch) for ch in experiment["name"])
    for rung_idx, train_fraction in enumerate(HYPERBAND_RUNG_TRAIN_FRACTIONS, start=1):
        stage_cfg = HYPERBAND_STAGE_SCHEDULE[rung_idx]
        rung_raw_rows = []
        stage_iterations = int(stage_cfg["iterations"])
        for stage_iter in range(1, stage_iterations + 1):
            iter_rows = []
            print(
                f"[Data-Hyperband:{MODEL_NAME}:{experiment['name']}] "
                f"rung={rung_idx} iter={stage_iter}/{stage_iterations} "
                f"configs={len(active_configs)} train_fraction={train_fraction:.2f}"
            )
            for config in active_configs:
                result = run_experiment(
                    experiment,
                    tables=tables,
                    split_patients=split_patients,
                    hyperparams=config["hyperparams"],
                    train_fraction=train_fraction,
                    train_subset_seed=subset_seed,
                    max_epochs=MLP_TRIAL_EPOCHS,
                    evaluate_test=False,
                    trial_id=config["config_id"],
                )
                metrics = result["val_metrics"]
                row = {
                    "model_name": MODEL_NAME,
                    "experiment": experiment["name"],
                    "rung": rung_idx,
                    "stage_iter": stage_iter,
                    "config_id": config["config_id"],
                    "origin": config.get("origin", ""),
                    "train_fraction": float(train_fraction),
                    "train_patients_used": result["split_sizes"]["train_patients"],
                    "train_patients_full": result["split_sizes"]["train_patients_full"],
                    "train_examples_used": result["split_sizes"]["train_examples"],
                    "train_examples_full": result["split_sizes"]["train_examples_full"],
                    "val_interval_mape": metrics.get("interval_mape", np.nan),
                    "val_scan_micro_auc": metrics.get("scan_micro_auc", np.nan),
                    "val_interval_mae": metrics.get("interval_mae", np.nan),
                    "val_interval_rmse": metrics.get("interval_rmse", np.nan),
                    "val_acc@0.5m": metrics.get("acc@0.5m", np.nan),
                    "val_acc@1m": metrics.get("acc@1m", np.nan),
                    "val_acc@2m": metrics.get("acc@2m", np.nan),
                    "hyperparams_json": hyperparams_to_json(config["hyperparams"]),
                }
                iter_rows.append(row)
                rung_raw_rows.append(row)
                all_rows.append(row)

            scored = compute_hyperband_scores(iter_rows)
            iter_unique_scored = dedupe_scored_configs(scored)
            iter_path = OUTPUT_DIR / f"data_hyperband_trials_{MODEL_NAME}_{experiment['name']}_rung_{rung_idx}_iter_{stage_iter}.csv"
            scored.to_csv(iter_path, index=False)
            append_report_section(
                f"hyperband::{MODEL_NAME}::{experiment['name']}::rung_{rung_idx}::iter_{stage_iter}",
                iter_unique_scored.head(10).to_string(index=False),
            )

            if stage_iter < stage_iterations:
                refine_n = min(int(stage_cfg["refine_top_k"]), len(iter_unique_scored))
                elite_n = min(int(stage_cfg["elite_keep"]), len(iter_unique_scored))
                refresh_count = int(stage_cfg["refresh_configs"])
                stage_search_space = make_refined_search_space(
                    iter_unique_scored.head(refine_n),
                    base_space=HYPERPARAM_SEARCH_SPACE,
                    expand_ratio=HYPERBAND_REFINE_EXPAND_RATIO,
                )
                elites = [
                    {"config_id": row["config_id"], "hyperparams": json.loads(row["hyperparams_json"]), "origin": "elite"}
                    for _, row in iter_unique_scored.head(elite_n).iterrows()
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

        rung_scored_all = dedupe_scored_configs(compute_hyperband_scores(rung_raw_rows))
        keep_n = min(int(stage_cfg["next_stage_keep"]), len(rung_scored_all))
        active_configs = [
            {
                "config_id": row["config_id"],
                "hyperparams": json.loads(row["hyperparams_json"]),
                "origin": "rung_survivor" if rung_idx < len(HYPERBAND_RUNG_TRAIN_FRACTIONS) else "final_best",
            }
            for _, row in rung_scored_all.head(keep_n).iterrows()
        ]
        rung_path = OUTPUT_DIR / f"data_hyperband_trials_{MODEL_NAME}_{experiment['name']}_rung_{rung_idx}.csv"
        rung_scored_all.to_csv(rung_path, index=False)
        rung_summaries.append({
            "model_name": MODEL_NAME,
            "experiment": experiment["name"],
            "rung": rung_idx,
            "stage_iterations": stage_iterations,
            "train_fraction": float(train_fraction),
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
        compute_hyperband_scores(all_df[all_df["rung"] == final_rung].to_dict("records")),
        key_col="hyperparams_json",
    )
    best_row = final_scored.iloc[0].to_dict()
    best_hp = json.loads(best_row["hyperparams_json"])
    return best_hp, best_row, pd.DataFrame(all_rows), pd.DataFrame(rung_summaries)


print("DATA_DIR:", DATA_DIR)
print("MODEL_NAME:", MODEL_NAME)
tables = load_tables()
target_patient_ids = tables["target"]["patient_id"].astype(str).unique()
shared_split_patients = make_patient_splits(target_patient_ids)
append_report_section(
    "run_setup",
    (
        f"model_name={MODEL_NAME}\n"
        f"data_dir={DATA_DIR}\n"
        f"selected_experiment={SELECTED_EXPERIMENT_NAME or 'ALL'}\n"
        f"train_patients={len(shared_split_patients['train'])}\n"
        f"val_patients={len(shared_split_patients['val'])}\n"
        f"test_patients={len(shared_split_patients['test'])}\n"
        f"hyperband_score={HYPERBAND_SCORE_NAME}"
    ),
)

experiment_results = []
summary_rows = []
metrics_rows = []
hyperband_trial_frames = []
hyperband_best_rows = []
hyperband_final_best_rows = []
hyperband_rung_frames = []
print(f"[Runner] Starting data-Hyperband for {len(EXPERIMENTS)} experiments...")
for exp_idx, experiment in enumerate(EXPERIMENTS, start=1):
    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) tuning {MODEL_NAME} {experiment['name']}")
    best_hp, best_row, trials_df, rung_df = hyperband_search(experiment, tables, shared_split_patients)
    hyperband_trial_frames.append(trials_df)
    hyperband_rung_frames.append(rung_df)
    hyperband_best_rows.append(best_row)

    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) final training with best hyperparameters for {MODEL_NAME} {experiment['name']}")
    result = run_experiment(
        experiment,
        tables=tables,
        split_patients=shared_split_patients,
        hyperparams=best_hp,
        train_fraction=1.0,
        train_subset_seed=HYPERBAND_RANDOM_SEED,
        max_epochs=MLP_FINAL_EPOCHS,
        evaluate_test=True,
        trial_id="best_final",
    )
    experiment_results.append(result)
    final_best_record = dict(best_row)
    final_best_record.update({f"test_{k}": v for k, v in result["test_metrics"].items()})
    final_best_record["final_train_epochs"] = 1
    final_best_record["final_best_val_loss"] = result["best_val_loss"]
    hyperband_final_best_rows.append(final_best_record)
    summary_rows.append({
        "model_name": MODEL_NAME,
        "experiment": result["name"],
        "feature_tables": ", ".join(result["feature_tables"]),
        "train_examples": result["split_sizes"]["train_examples"],
        "val_examples": result["split_sizes"]["val_examples"],
        "test_examples": result["split_sizes"]["test_examples"],
        "best_hyperband_config_id": best_row["config_id"],
        "best_hyperband_score": best_row["hyperband_score"],
    })
    metrics_rows.append({"model_name": MODEL_NAME, "experiment": result["name"], **result["test_metrics"]})
    print(f"[Runner] ({exp_idx}/{len(EXPERIMENTS)}) finished {MODEL_NAME} {experiment['name']}")

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
    save_dataframe_artifact(result["history"], f"training_history_{MODEL_NAME}_{result['name']}", title=f"training_history::{MODEL_NAME}::{result['name']}", index=False)
    detail_df = build_prediction_detail_df(result["pred"], result["interval_thresholds"])
    detail_csv_path, detail_txt_path = save_dataframe_artifact(
        detail_df,
        f"prediction_detail_{MODEL_NAME}_{result['name']}",
        title=f"prediction_detail::{MODEL_NAME}::{result['name']}",
        index=False,
    )
    incorrect_df = build_incorrect_prediction_df(detail_df, result["test_examples_df"], result["name"])
    incorrect_csv_path, incorrect_txt_path = save_dataframe_artifact(
        incorrect_df,
        f"incorrect_predictions_{MODEL_NAME}_{result['name']}",
        title=f"incorrect_predictions::{MODEL_NAME}::{result['name']}",
        index=False,
    )
    metrics_df = pd.DataFrame([{"model_name": MODEL_NAME, "experiment": result["name"], **result["test_metrics"]}])
    metrics_csv_path, metrics_txt_path = save_dataframe_artifact(
        metrics_df,
        f"experiment_metrics_{MODEL_NAME}_{result['name']}",
        title=f"experiment_metrics::{MODEL_NAME}::{result['name']}",
        index=False,
    )
    prediction_artifact_rows.append({
        "model_name": MODEL_NAME,
        "experiment": result["name"],
        "feature_tables": ", ".join(result["feature_tables"]),
        "prediction_detail_csv": str(detail_csv_path),
        "prediction_detail_txt": str(detail_txt_path),
        "incorrect_prediction_csv": str(incorrect_csv_path),
        "incorrect_prediction_txt": str(incorrect_txt_path),
        "metrics_csv": str(metrics_csv_path),
        "metrics_txt": str(metrics_txt_path),
        "n_rows": len(detail_df),
        "n_incorrect": len(incorrect_df),
    })

summary_df = pd.DataFrame(summary_rows)
metrics_summary_df = pd.DataFrame(metrics_rows)
prediction_artifacts_df = pd.DataFrame(prediction_artifact_rows)
save_dataframe_artifact(summary_df, "experiment_summary", title="experiment_summary", index=False)
save_dataframe_artifact(metrics_summary_df, "experiment_metrics_summary", title="experiment_metrics_summary", index=False)
save_dataframe_artifact(prediction_artifacts_df, "prediction_artifacts", title="prediction_artifacts", index=False)

metadata = {
    "model_name": MODEL_NAME,
    "scan_type_names": SCAN_TYPE_NAMES,
    "interval_metric_names": ["MAE", "RMSE", "MAPE"],
    "interval_acc_thresholds": INTERVAL_ACC_THRESHOLDS,
    "scan_threshold": SCAN_THRESHOLD,
    "hyperband_score": HYPERBAND_SCORE_NAME,
    "hyperband_score_formula": "val_scan_micro_auc - ((val_interval_mape / 100.0) / (1.0 + (val_interval_mape / 100.0)))",
    "hyperband_resource": "patient_wise_train_fraction",
    "hyperband_rung_train_fractions": HYPERBAND_RUNG_TRAIN_FRACTIONS,
    "hyperband_initial_configs": HYPERBAND_INITIAL_CONFIGS,
    "hyperband_stage_schedule": HYPERBAND_STAGE_SCHEDULE,
    "hyperband_refine_expand_ratio": HYPERBAND_REFINE_EXPAND_RATIO,
    "hyperband_unique_config_key": "hyperparams_json",
    "data_dir": str(DATA_DIR),
    "output_dir": str(OUTPUT_DIR),
    "model_default_hyperparams": LR_DEFAULT_HYPERPARAMS,
    "ridge_default_hyperparams": RIDGE_DEFAULT_HYPERPARAMS,
    "model_search_space": LR_HYPERPARAM_SEARCH_SPACE,
    "ridge_search_space": RIDGE_HYPERPARAM_SEARCH_SPACE,
    "search_space": HYPERPARAM_SEARCH_SPACE,
}
with open(OUTPUT_DIR / "plot_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
append_report_section("saved_plot_metadata", str(OUTPUT_DIR / "plot_metadata.json"))

display(summary_df)
display(metrics_summary_df)
display(hyperband_best_df)
display(hyperband_final_best_df)
display(prediction_artifacts_df)


