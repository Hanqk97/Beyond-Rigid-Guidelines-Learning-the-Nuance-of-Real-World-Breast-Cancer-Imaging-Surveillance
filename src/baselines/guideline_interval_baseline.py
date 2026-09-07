#!/usr/bin/env python
"""Guideline interval baseline for MAMMO/MRI/ULTRASOUND surveillance.

Rules:
* MAMMO_TOMO -> MAMMO_TOMO: 12 months
* MRI -> MRI: 12 months
* MAMMO_TOMO -> MRI or MRI -> MAMMO_TOMO: 6 months
* ULTRASOUND -> ULTRASOUND: nearest allowed guideline interval among 6, 12, 24 months

The ultrasound rule is an oracle-binned guideline baseline because it maps the
observed interval to the closest allowed guideline interval.
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
GUIDELINE_SCAN_TYPES = ["MAMMO_TOMO", "MRI", "ULTRASOUND"]
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


def make_patient_splits(patient_ids: pd.Series | np.ndarray) -> dict[str, set[str]]:
    patient_ids = sorted(pd.Series(patient_ids).astype(str).unique())
    if len(patient_ids) < 3:
        raise ValueError(f"Need at least 3 patients to create train/val/test splits, got {len(patient_ids)}")

    def sklearn_style_split(values: list[str], test_size: float) -> tuple[list[str], list[str]]:
        n_test = int(np.ceil(len(values) * float(test_size)))
        indices = np.random.RandomState(SEED).permutation(len(values))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        return [values[idx] for idx in train_indices], [values[idx] for idx in test_indices]

    train_patients, temp_patients = sklearn_style_split(patient_ids, test_size=0.30)
    val_patients, test_patients = sklearn_style_split(temp_patients, test_size=0.50)
    return {"train": set(train_patients), "val": set(val_patients), "test": set(test_patients)}


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


def active_guideline_types(row: pd.Series) -> list[str]:
    return [scan_name for scan_name in GUIDELINE_SCAN_TYPES if int(row[f"has_{scan_name}"]) == 1]


def guideline_interval(anchor_type: str, target_type: str, true_interval: float) -> float | None:
    if anchor_type == "MAMMO_TOMO" and target_type == "MAMMO_TOMO":
        return 12.0
    if anchor_type == "MRI" and target_type == "MRI":
        return 12.0
    if {anchor_type, target_type} == {"MAMMO_TOMO", "MRI"}:
        return 6.0
    if anchor_type == "ULTRASOUND" and target_type == "ULTRASOUND":
        choices = np.asarray([6.0, 12.0, 24.0], dtype=float)
        return float(choices[np.argmin(np.abs(choices - float(true_interval)))])
    return None


def build_guideline_rows(df: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    rows = []
    split_df = df[df["patient_id"].isin(patients)].copy()
    for patient_id, group in split_df.groupby("patient_id", sort=False):
        group = group.sort_values(["report_date", "sequence"], kind="mergesort").reset_index(drop=True)
        for anchor_idx in range(len(group) - 1):
            anchor = group.iloc[anchor_idx]
            anchor_types = active_guideline_types(anchor)
            if not anchor_types:
                continue

            future = group.iloc[anchor_idx + 1 :].copy()
            for anchor_type in anchor_types:
                if anchor_type in {"MAMMO_TOMO", "MRI"}:
                    candidate_types = ["MAMMO_TOMO", "MRI"]
                else:
                    candidate_types = ["ULTRASOUND"]

                hit = None
                hit_type = None
                for _, future_row in future.iterrows():
                    active = active_guideline_types(future_row)
                    matched = [scan_name for scan_name in candidate_types if scan_name in active]
                    if matched:
                        hit = future_row
                        hit_type = matched[0]
                        break

                if hit is None or hit_type is None:
                    continue

                true_interval = float((hit["report_date"] - anchor["report_date"]).days) / DAYS_PER_MONTH
                if true_interval < 0:
                    continue
                pred_interval = guideline_interval(anchor_type, hit_type, true_interval)
                if pred_interval is None:
                    continue

                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "anchor_scan_type": anchor_type,
                        "target_scan_type": hit_type,
                        "guideline_rule": f"{anchor_type}_to_{hit_type}",
                        "anchor_report_date": anchor["report_date"],
                        "target_report_date": hit["report_date"],
                        "anchor_sequence": int(anchor["sequence"]),
                        "target_sequence": int(hit["sequence"]),
                        "true_interval_months": true_interval,
                        "pred_interval_months": pred_interval,
                    }
                )
    return pd.DataFrame(rows)


def add_detail_columns(rows: pd.DataFrame) -> pd.DataFrame:
    detail = rows.copy()
    detail["interval_error_months"] = detail["pred_interval_months"] - detail["true_interval_months"]
    detail["interval_abs_error_months"] = detail["interval_error_months"].abs()
    for threshold in INTERVAL_ACC_THRESHOLDS:
        detail[f"within_{threshold:g}m"] = detail["interval_abs_error_months"] <= threshold

    for scan_name in SCAN_TYPE_NAMES:
        is_target = (detail["target_scan_type"] == scan_name).astype(int)
        detail[f"y_scan_{scan_name}"] = is_target
        detail[f"p_scan_{scan_name}"] = is_target.astype(float)
        detail[f"pred_scan_{scan_name}"] = is_target

    columns = [
        "patient_id",
        "anchor_scan_type",
        "target_scan_type",
        "guideline_rule",
        "anchor_report_date",
        "target_report_date",
        "anchor_sequence",
        "target_sequence",
        "true_interval_months",
        "pred_interval_months",
        "interval_error_months",
        "interval_abs_error_months",
        *[f"within_{threshold:g}m" for threshold in INTERVAL_ACC_THRESHOLDS],
        *sum(([f"y_scan_{name}", f"p_scan_{name}", f"pred_scan_{name}"] for name in SCAN_TYPE_NAMES), []),
    ]
    return detail[columns].reset_index(drop=True)


def summarize_metrics(detail: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    groups = [(None, detail)] if group_col is None else list(detail.groupby(group_col, sort=False))
    rows = []
    for key, part in groups:
        true_months = part["true_interval_months"].to_numpy(dtype=float)
        pred_months = part["pred_interval_months"].to_numpy(dtype=float)
        abs_error = np.abs(pred_months - true_months)
        nonzero = true_months != 0
        row = {
            "experiment": "guideline_interval",
            "n_test_intervals": int(len(part)),
            "interval_mae": float(abs_error.mean()) if len(part) else np.nan,
            "interval_rmse": float(np.sqrt(np.mean((pred_months - true_months) ** 2))) if len(part) else np.nan,
            "interval_mape": float(100.0 * np.mean(abs_error[nonzero] / true_months[nonzero])) if nonzero.any() else np.nan,
        }
        if group_col is not None:
            row[group_col] = key
            row["pred_interval_mean_months"] = float(part["pred_interval_months"].mean()) if len(part) else np.nan
            row["true_interval_mean_months"] = float(part["true_interval_months"].mean()) if len(part) else np.nan
        for threshold in INTERVAL_ACC_THRESHOLDS:
            row[f"acc@{threshold:g}m"] = float((abs_error <= threshold).mean()) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_data_dir = project_root / "data" / "prepared" / "multitask"
    default_output_dir = Path(__file__).resolve().parent / "outputs" / "guideline_interval"
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
    test_rows = build_guideline_rows(imaging, split_patients["test"])
    if test_rows.empty:
        raise ValueError("No guideline-eligible test intervals were found.")
    detail = add_detail_columns(test_rows)
    metrics = summarize_metrics(detail)
    metrics_by_rule = summarize_metrics(detail, group_col="guideline_rule")
    metrics_by_target = summarize_metrics(detail, group_col="target_scan_type")
    summary = pd.DataFrame(
        [
            {
                "train_patients": len(split_patients["train"]),
                "val_patients": len(split_patients["val"]),
                "test_patients": len(split_patients["test"]),
                "test_guideline_intervals": len(detail),
            }
        ]
    )

    detail.to_csv(output_dir / "prediction_detail_guideline_interval.csv", index=False)
    metrics.to_csv(output_dir / "experiment_metrics_guideline_interval.csv", index=False)
    metrics_by_rule.to_csv(output_dir / "experiment_metrics_by_rule_guideline_interval.csv", index=False)
    metrics_by_target.to_csv(output_dir / "experiment_metrics_by_target_scan_type_guideline_interval.csv", index=False)
    summary.to_csv(output_dir / "experiment_summary_guideline_interval.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "seed": SEED,
                "scan_type_names": SCAN_TYPE_NAMES,
                "guideline_scan_types": GUIDELINE_SCAN_TYPES,
                "interval_acc_thresholds": INTERVAL_ACC_THRESHOLDS,
                "rules": {
                    "MAMMO_TOMO_to_MAMMO_TOMO": 12,
                    "MRI_to_MRI": 12,
                    "MAMMO_TOMO_to_MRI": 6,
                    "MRI_to_MAMMO_TOMO": 6,
                    "ULTRASOUND_to_ULTRASOUND": "nearest of 6, 12, 24 using observed interval",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = "\n\n".join(
        [
            "# Guideline interval baseline",
            summary.to_string(index=False),
            "## Overall metrics\n" + metrics.to_string(index=False),
            "## Metrics by rule\n" + metrics_by_rule.to_string(index=False),
            "## Metrics by target scan type\n" + metrics_by_target.to_string(index=False),
        ]
    )
    (output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")

    print(f"Wrote guideline interval baseline outputs to: {output_dir}")
    print(metrics.to_string(index=False))
    print(metrics_by_rule.to_string(index=False))


if __name__ == "__main__":
    main()
