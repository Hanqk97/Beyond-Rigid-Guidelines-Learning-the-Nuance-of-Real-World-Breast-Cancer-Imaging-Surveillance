#!/usr/bin/env python
"""Build restricted matched-baseline inputs used by the paper aggregation step."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
DAYS_PER_MONTH = 30.4375
SCAN_TYPES = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
GUIDELINE_TYPES = {"MAMMO_TOMO", "MRI", "ULTRASOUND"}


def parse_scan_vector(value) -> list[int]:
    raw = list(value) if isinstance(value, (list, tuple, np.ndarray)) else ast.literal_eval(str(value))
    parsed = [int(float(item) >= 0.5) for item in raw]
    if len(parsed) != len(SCAN_TYPES):
        raise ValueError(f"Expected {len(SCAN_TYPES)} scan flags, got {len(parsed)}")
    return parsed


def active_types(value, allowed=None) -> list[str]:
    names = [scan for scan, flag in zip(SCAN_TYPES, parse_scan_vector(value)) if flag]
    return names if allowed is None else [name for name in names if name in allowed]


def test_patient_ids(patient_ids: pd.Series) -> set[str]:
    patients = sorted(pd.Series(patient_ids).astype(str).unique())
    _, temporary = train_test_split(patients, test_size=0.30, random_state=SEED)
    _, test = train_test_split(temporary, test_size=0.50, random_state=SEED)
    return set(test)


def build_test_examples(data_dir: Path) -> pd.DataFrame:
    imaging = pd.read_csv(data_dir / "imaging_timeseries.csv", dtype={"patient_id": "string"})
    imaging["patient_id"] = imaging["patient_id"].astype(str)
    imaging["report_date"] = pd.to_datetime(imaging["report_date"], errors="coerce")
    imaging["sequence"] = pd.to_numeric(imaging["sequence"], errors="coerce").fillna(-1).astype("int64")
    imaging = imaging.loc[imaging["report_date"].notna()].sort_values(
        ["patient_id", "report_date", "sequence"], kind="mergesort"
    )
    rows = []
    for patient_id, group in imaging.groupby("patient_id", sort=False):
        group = group.reset_index(drop=True)
        for target_index in range(1, len(group)):
            anchor = group.iloc[target_index - 1]
            target = group.iloc[target_index]
            rows.append({
                "patient_id": str(patient_id),
                "anchor_report_date": anchor["report_date"],
                "target_report_date": target["report_date"],
                "anchor_sequence": int(anchor["sequence"]),
                "target_sequence": int(target["sequence"]),
                "anchor_scan_vector": anchor["scan_type_vector"],
                "target_scan_vector": target["scan_type_vector"],
                "anchor_guideline_types": active_types(anchor["scan_type_vector"], GUIDELINE_TYPES),
                "target_guideline_types": active_types(target["scan_type_vector"], GUIDELINE_TYPES),
                "target_scan_types": active_types(target["scan_type_vector"]),
                "true_interval_months_reconstructed": (
                    target["report_date"] - anchor["report_date"]
                ).days / DAYS_PER_MONTH,
            })
    examples = pd.DataFrame(rows)
    return examples.loc[examples["patient_id"].isin(test_patient_ids(examples["patient_id"]))].reset_index(drop=True)


def find_gru_run(results_dir: Path, setting: str) -> Path:
    pattern = f"gru_tree_multitask_compare_next_scan_data_hyperband_normalized_score_*_{setting}_pid*"
    candidates = sorted(results_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    candidates = [path for path in candidates if (path / f"prediction_detail_{setting}.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No GRU-{setting} prediction output found under {results_dir}")
    return candidates[0]


def align_predictions(examples: pd.DataFrame, prediction_path: Path) -> pd.DataFrame:
    prediction = pd.read_csv(prediction_path, dtype={"patient_id": "string"})
    prediction["patient_id"] = prediction["patient_id"].astype(str)
    if len(examples) != len(prediction):
        raise ValueError(f"Row mismatch: reconstructed={len(examples):,}, predictions={len(prediction):,}")
    if not np.array_equal(examples["patient_id"].to_numpy(), prediction["patient_id"].to_numpy()):
        raise ValueError("Patient order differs between reconstructed examples and predictions")
    difference = np.abs(
        examples["true_interval_months_reconstructed"].to_numpy(float)
        - prediction["true_interval_months"].to_numpy(float)
    )
    if difference.max(initial=0.0) > 1e-5:
        raise ValueError(f"Interval alignment failed; maximum difference={difference.max():.8f}")
    return pd.concat(
        [examples, prediction[["true_interval_months", "pred_interval_months", "interval_abs_error_months"]]],
        axis=1,
    )


def guideline_interval(anchor_type: str, target_type: str, true_interval: float):
    if anchor_type == target_type and anchor_type in {"MAMMO_TOMO", "MRI"}:
        return 12.0
    if {anchor_type, target_type} == {"MAMMO_TOMO", "MRI"}:
        return 6.0
    if anchor_type == target_type == "ULTRASOUND":
        choices = np.asarray([6.0, 12.0, 24.0])
        return float(choices[np.argmin(np.abs(choices - true_interval))])
    return None


def summarize_errors(detail: pd.DataFrame, baseline_error: str, model_error: str, group: str):
    def metrics(part):
        baseline = part[baseline_error].to_numpy(float)
        model = part[model_error].to_numpy(float)
        return {
            "n": len(part),
            "baseline_mae": baseline.mean(),
            "model_mae": model.mean(),
            "baseline_rmse": np.sqrt(np.mean(np.square(baseline))),
            "model_rmse": np.sqrt(np.mean(np.square(model))),
        }

    rows = [{group: str(name), **metrics(part)} for name, part in detail.groupby(group, observed=True, sort=False)]
    rows.append({group: "Overall", **metrics(detail)})
    return pd.DataFrame(rows)


def build_guideline(aligned: pd.DataFrame, output_dir: Path, model_display: str) -> None:
    rows = []
    for row in aligned.itertuples(index=False):
        for anchor_type in row.anchor_guideline_types:
            for target_type in row.target_guideline_types:
                prediction = guideline_interval(anchor_type, target_type, float(row.true_interval_months))
                if prediction is None:
                    continue
                rule = f"{anchor_type}_to_{target_type}"
                guideline_error = abs(prediction - float(row.true_interval_months))
                model_error = abs(float(row.pred_interval_months) - float(row.true_interval_months))
                rows.append({
                    "patient_id": row.patient_id,
                    "guideline_rule": rule,
                    "transition": rule.replace("_to_", " -> "),
                    "anchor_report_date": row.anchor_report_date,
                    "target_report_date": row.target_report_date,
                    "true_interval_months": float(row.true_interval_months),
                    "guideline_pred_interval_months": prediction,
                    "model_pred_interval_months": float(row.pred_interval_months),
                    "guideline_abs_error_months": guideline_error,
                    "model_abs_error_months": model_error,
                    "model_closer": model_error < guideline_error,
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No guideline-covered transitions were found")
    rule_order = [
        "MRI_to_MRI", "MRI_to_MAMMO_TOMO", "MAMMO_TOMO_to_MRI",
        "MAMMO_TOMO_to_MAMMO_TOMO", "ULTRASOUND_to_ULTRASOUND",
    ]
    detail["guideline_rule"] = pd.Categorical(detail["guideline_rule"], rule_order, ordered=True)
    detail = detail.sort_values(["guideline_rule", "patient_id", "anchor_report_date"])
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / f"matched_guideline_detail_{model_display}.csv", index=False)
    summary = summarize_errors(
        detail, "guideline_abs_error_months", "model_abs_error_months", "transition"
    ).rename(columns={
        "baseline_mae": "guideline_mae", "baseline_rmse": "guideline_rmse",
    })
    summary.to_csv(output_dir / f"matched_guideline_summary_wide_{model_display}.csv", index=False)


def build_average(
    aligned: pd.DataFrame, average_detail_path: Path, output_dir: Path, model_display: str
) -> None:
    model_events = aligned.explode("target_scan_types").rename(columns={"target_scan_types": "scan_type"})
    model_events["match_key"] = (
        model_events["patient_id"].astype(str)
        + "|" + model_events["anchor_report_date"].dt.strftime("%Y-%m-%d")
        + "|" + model_events["target_report_date"].dt.strftime("%Y-%m-%d")
        + "|" + model_events["scan_type"].astype(str)
    )
    model_match = model_events.rename(columns={
        "pred_interval_months": "model_pred_interval_months",
        "interval_abs_error_months": "model_abs_error_months",
    })[["match_key", "model_pred_interval_months", "model_abs_error_months"]]

    average = pd.read_csv(average_detail_path, dtype={"patient_id": "string"})
    average["patient_id"] = average["patient_id"].astype(str)
    average["anchor_report_date"] = pd.to_datetime(average["anchor_report_date"])
    average["next_same_type_report_date"] = pd.to_datetime(average["next_same_type_report_date"])
    average["match_key"] = (
        average["patient_id"]
        + "|" + average["anchor_report_date"].dt.strftime("%Y-%m-%d")
        + "|" + average["next_same_type_report_date"].dt.strftime("%Y-%m-%d")
        + "|" + average["scan_type"].astype(str)
    )
    matched = average.merge(model_match, on="match_key", how="inner", validate="one_to_one")
    matched = matched.rename(columns={
        "pred_interval_months": "average_pred_interval_months",
        "interval_abs_error_months": "average_abs_error_months",
    })
    if matched.empty:
        raise ValueError("No average-baseline rows matched GRU-DSIMB predictions")
    matched["model_closer"] = matched["model_abs_error_months"] < matched["average_abs_error_months"]
    scan_order = ["CT", "MRI", "PET", "ULTRASOUND", "XRAY", "MAMMO_TOMO", "NUCLEAR_MED"]
    matched["scan_type"] = pd.Categorical(matched["scan_type"], scan_order, ordered=True)
    matched = matched.sort_values(["scan_type", "patient_id", "anchor_report_date"])
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / f"matched_average_detail_{model_display}.csv", index=False)
    summary = summarize_errors(
        matched, "average_abs_error_months", "model_abs_error_months", "scan_type"
    ).rename(columns={
        "baseline_mae": "average_mae", "baseline_rmse": "average_rmse",
    })
    summary.to_csv(output_dir / f"matched_average_summary_wide_{model_display}.csv", index=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data" / "prepared" / "multitask")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "restricted_artifacts" / "results")
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Destination root for matched directories; defaults to --results-dir.",
    )
    parser.add_argument(
        "--average-baseline-dir", type=Path,
        default=repo_root / "restricted_artifacts" / "baselines" / "average_same_type_interval",
    )
    parser.add_argument("--setting", default="DSIMB")
    args = parser.parse_args()

    run_dir = find_gru_run(args.results_dir.resolve(), args.setting)
    examples = build_test_examples(args.data_dir.resolve())
    aligned = align_predictions(examples, run_dir / f"prediction_detail_{args.setting}.csv")
    display_name = f"GRU-{args.setting}"
    output_root = (args.output_root or args.results_dir).resolve()
    build_guideline(aligned, output_root / "guideline_matched_comparison", display_name)
    build_average(
        aligned,
        args.average_baseline_dir / "prediction_detail_average_same_type_interval.csv",
        output_root / "average_matched_comparison",
        display_name,
    )
    print(f"Matched baseline inputs written under: {output_root}")


if __name__ == "__main__":
    main()
