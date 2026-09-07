#!/usr/bin/env python
"""Average same-type scan interval baseline.

This baseline estimates the expected return interval for each scan type from
training patients only. For a test anchor scan, it predicts the train-set mean
interval until the next scan of the same type, ignoring intervening scan types.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
SCAN_TYPE_NAMES = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
INTERVAL_ACC_THRESHOLDS = [0.5, 1.0, 2.0]
DAYS_PER_MONTH = 30.4375


def parse_scan_vector(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = ast.literal_eval(str(value))
    out = [int(float(x) >= 0.5) for x in raw]
    if len(out) != len(SCAN_TYPE_NAMES):
        raise ValueError(f"Expected {len(SCAN_TYPE_NAMES)} scan flags, got {len(out)}: {value!r}")
    return out


def months_between(start: object, end: object) -> float:
    return float((pd.Timestamp(end) - pd.Timestamp(start)).days) / DAYS_PER_MONTH


def make_patient_splits(patient_ids: pd.Series | np.ndarray) -> dict[str, set[str]]:
    patient_ids = sorted(pd.Series(patient_ids).astype(str).unique())
    if len(patient_ids) < 3:
        raise ValueError(f"Need at least 3 patients to create train/val/test splits, got {len(patient_ids)}")

    def sklearn_style_split(values: list[str], test_size: float) -> tuple[list[str], list[str]]:
        # Mirrors sklearn.model_selection.train_test_split(..., shuffle=True,
        # random_state=SEED) closely enough for this deterministic baseline,
        # without importing sklearn/Intel runtime just to split IDs.
        n_test = int(np.ceil(len(values) * float(test_size)))
        indices = np.random.RandomState(SEED).permutation(len(values))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        train_values = [values[idx] for idx in train_indices]
        test_values = [values[idx] for idx in test_indices]
        return train_values, test_values

    train_patients, temp_patients = sklearn_style_split(patient_ids, test_size=0.30)
    val_patients, test_patients = sklearn_style_split(temp_patients, test_size=0.50)
    return {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }


def load_imaging_table(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "imaging_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing imaging table: {path}")

    df = pd.read_csv(path, dtype={"patient_id": "string"})
    required = {"patient_id", "report_date", "sequence", "scan_type_vector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["sequence"] = pd.to_numeric(df["sequence"], errors="coerce").fillna(-1).astype("int64")
    df = df[df["report_date"].notna()].copy()
    vectors = df["scan_type_vector"].apply(parse_scan_vector)
    for idx, scan_name in enumerate(SCAN_TYPE_NAMES):
        df[f"has_{scan_name}"] = vectors.apply(lambda row, i=idx: int(row[i]))
    return df.sort_values(["patient_id", "report_date", "sequence"], kind="mergesort").reset_index(drop=True)


def same_type_interval_rows(df: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    split_df = df[df["patient_id"].isin(patients)].copy()
    if split_df.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "scan_type",
                "anchor_report_date",
                "next_same_type_report_date",
                "true_interval_months",
                "anchor_sequence",
                "next_same_type_sequence",
            ]
        )

    value_cols = [f"has_{scan_name}" for scan_name in SCAN_TYPE_NAMES]
    long_df = split_df.melt(
        id_vars=["patient_id", "report_date", "sequence"],
        value_vars=value_cols,
        var_name="scan_type",
        value_name="has_scan",
    )
    long_df = long_df[long_df["has_scan"] == 1].copy()
    long_df["scan_type"] = long_df["scan_type"].str.replace("has_", "", regex=False)
    long_df = long_df.sort_values(["patient_id", "scan_type", "report_date", "sequence"], kind="mergesort")
    grouped = long_df.groupby(["patient_id", "scan_type"], sort=False)
    long_df["next_same_type_report_date"] = grouped["report_date"].shift(-1)
    long_df["next_same_type_sequence"] = grouped["sequence"].shift(-1)
    long_df = long_df[long_df["next_same_type_report_date"].notna()].copy()
    long_df["true_interval_months"] = (
        (long_df["next_same_type_report_date"] - long_df["report_date"]).dt.days.astype(float) / DAYS_PER_MONTH
    )
    long_df = long_df[long_df["true_interval_months"] >= 0].copy()
    long_df = long_df.rename(
        columns={
            "report_date": "anchor_report_date",
            "sequence": "anchor_sequence",
        }
    )
    long_df["anchor_sequence"] = long_df["anchor_sequence"].astype(int)
    long_df["next_same_type_sequence"] = long_df["next_same_type_sequence"].astype(int)
    return long_df[
        [
            "patient_id",
            "scan_type",
            "anchor_report_date",
            "next_same_type_report_date",
            "true_interval_months",
            "anchor_sequence",
            "next_same_type_sequence",
        ]
    ].reset_index(drop=True)


def compute_train_means(train_rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if train_rows.empty:
        raise ValueError("No train same-type intervals were found.")

    global_mean = float(train_rows["true_interval_months"].mean())
    means = (
        train_rows.groupby("scan_type", as_index=False)
        .agg(
            mean_interval_months=("true_interval_months", "mean"),
            median_interval_months=("true_interval_months", "median"),
            n_intervals=("true_interval_months", "size"),
        )
        .sort_values("scan_type")
        .reset_index(drop=True)
    )
    means["fallback_used"] = False

    observed = set(means["scan_type"])
    missing_rows = [
        {
            "scan_type": scan_name,
            "mean_interval_months": global_mean,
            "median_interval_months": np.nan,
            "n_intervals": 0,
            "fallback_used": True,
        }
        for scan_name in SCAN_TYPE_NAMES
        if scan_name not in observed
    ]
    if missing_rows:
        means = pd.concat([means, pd.DataFrame(missing_rows)], ignore_index=True)

    return means.sort_values("scan_type").reset_index(drop=True), global_mean


def build_prediction_detail(test_rows: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    mean_map = dict(zip(means["scan_type"], means["mean_interval_months"]))
    detail = test_rows.copy()
    detail["pred_interval_months"] = detail["scan_type"].map(mean_map).astype(float)
    detail["interval_error_months"] = detail["pred_interval_months"] - detail["true_interval_months"]
    detail["interval_abs_error_months"] = detail["interval_error_months"].abs()
    for threshold in INTERVAL_ACC_THRESHOLDS:
        detail[f"within_{threshold:g}m"] = detail["interval_abs_error_months"] <= threshold

    for scan_name in SCAN_TYPE_NAMES:
        is_type = (detail["scan_type"] == scan_name).astype(int)
        detail[f"y_scan_{scan_name}"] = is_type
        detail[f"p_scan_{scan_name}"] = is_type.astype(float)
        detail[f"pred_scan_{scan_name}"] = is_type

    columns = [
        "patient_id",
        "scan_type",
        "anchor_report_date",
        "next_same_type_report_date",
        "anchor_sequence",
        "next_same_type_sequence",
        "true_interval_months",
        "pred_interval_months",
        "interval_error_months",
        "interval_abs_error_months",
        *[f"within_{threshold:g}m" for threshold in INTERVAL_ACC_THRESHOLDS],
        *sum(
            ([f"y_scan_{name}", f"p_scan_{name}", f"pred_scan_{name}"] for name in SCAN_TYPE_NAMES),
            [],
        ),
    ]
    return detail[columns].reset_index(drop=True)


def compute_interval_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    true_months = detail["true_interval_months"].to_numpy(dtype=float)
    pred_months = detail["pred_interval_months"].to_numpy(dtype=float)
    abs_error = np.abs(pred_months - true_months)
    nonzero = true_months != 0
    metrics = {
        "experiment": "average_same_type_interval",
        "interval_mae": float(abs_error.mean()) if len(abs_error) else np.nan,
        "interval_rmse": float(np.sqrt(np.mean((pred_months - true_months) ** 2))) if len(abs_error) else np.nan,
        "interval_mape": float(100.0 * np.mean(abs_error[nonzero] / true_months[nonzero])) if nonzero.any() else np.nan,
    }
    for threshold in INTERVAL_ACC_THRESHOLDS:
        metrics[f"acc@{threshold:g}m"] = float((abs_error <= threshold).mean()) if len(abs_error) else np.nan
    return pd.DataFrame([metrics])


def compute_interval_metrics_by_scan_type(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scan_name in SCAN_TYPE_NAMES:
        part = detail[detail["scan_type"] == scan_name]
        if part.empty:
            row = {
                "scan_type": scan_name,
                "n_test_intervals": 0,
                "pred_interval_months": np.nan,
                "true_interval_mean_months": np.nan,
                "true_interval_median_months": np.nan,
                "interval_mae": np.nan,
                "interval_rmse": np.nan,
                "interval_mape": np.nan,
            }
            for threshold in INTERVAL_ACC_THRESHOLDS:
                row[f"acc@{threshold:g}m"] = np.nan
            rows.append(row)
            continue

        true_months = part["true_interval_months"].to_numpy(dtype=float)
        pred_months = part["pred_interval_months"].to_numpy(dtype=float)
        abs_error = np.abs(pred_months - true_months)
        nonzero = true_months != 0
        row = {
            "scan_type": scan_name,
            "n_test_intervals": int(len(part)),
            "pred_interval_months": float(part["pred_interval_months"].iloc[0]),
            "true_interval_mean_months": float(np.mean(true_months)),
            "true_interval_median_months": float(np.median(true_months)),
            "interval_mae": float(abs_error.mean()),
            "interval_rmse": float(np.sqrt(np.mean((pred_months - true_months) ** 2))),
            "interval_mape": float(100.0 * np.mean(abs_error[nonzero] / true_months[nonzero])) if nonzero.any() else np.nan,
        }
        for threshold in INTERVAL_ACC_THRESHOLDS:
            row[f"acc@{threshold:g}m"] = float((abs_error <= threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    data_dir: Path,
    split_sizes: dict[str, int],
    global_mean: float,
    means: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_by_type: pd.DataFrame,
) -> None:
    lines = [
        "# Average same-type interval baseline",
        "",
        f"data_dir: {data_dir}",
        f"seed: {SEED}",
        "",
        "## Split sizes",
        *(f"{key}: {value}" for key, value in split_sizes.items()),
        "",
        "## Prediction rule",
        "For each anchor scan type, predict the train-set mean interval until the next scan of the same type.",
        "Intervening scan types are ignored.",
        f"global_fallback_mean_interval_months: {global_mean:.6f}",
        "",
        "## Metrics",
        metrics.to_string(index=False),
        "",
        "## Metrics by scan type",
        metrics_by_type.to_string(index=False),
        "",
        "## Train means by scan type",
        means.to_string(index=False),
        "",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_data_dir = project_root / "data" / "prepared" / "multitask"
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "average_same_type_interval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    imaging = load_imaging_table(data_dir)
    split_patients = make_patient_splits(imaging["patient_id"])
    train_rows = same_type_interval_rows(imaging, split_patients["train"])
    val_rows = same_type_interval_rows(imaging, split_patients["val"])
    test_rows = same_type_interval_rows(imaging, split_patients["test"])
    means, global_mean = compute_train_means(train_rows)
    detail = build_prediction_detail(test_rows, means)
    metrics = compute_interval_metrics(detail)
    metrics_by_type = compute_interval_metrics_by_scan_type(detail)

    split_sizes = {
        "train_patients": len(split_patients["train"]),
        "val_patients": len(split_patients["val"]),
        "test_patients": len(split_patients["test"]),
        "train_same_type_intervals": len(train_rows),
        "val_same_type_intervals": len(val_rows),
        "test_same_type_intervals": len(test_rows),
    }
    summary = pd.DataFrame([{**split_sizes, "global_mean_interval_months": global_mean}])

    means.to_csv(output_dir / "train_same_type_mean_intervals.csv", index=False)
    detail.to_csv(output_dir / "prediction_detail_average_same_type_interval.csv", index=False)
    metrics.to_csv(output_dir / "experiment_metrics_average_same_type_interval.csv", index=False)
    metrics_by_type.to_csv(output_dir / "experiment_metrics_by_scan_type_average_same_type_interval.csv", index=False)
    summary.to_csv(output_dir / "experiment_summary_average_same_type_interval.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "seed": SEED,
                "scan_type_names": SCAN_TYPE_NAMES,
                "interval_acc_thresholds": INTERVAL_ACC_THRESHOLDS,
                "task": "next same-type scan interval; intervening scan types ignored",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, data_dir, split_sizes, global_mean, means, metrics, metrics_by_type)

    print(f"Wrote average same-type interval baseline outputs to: {output_dir}")
    print(metrics.to_string(index=False))
    print(metrics_by_type.to_string(index=False))


if __name__ == "__main__":
    main()
