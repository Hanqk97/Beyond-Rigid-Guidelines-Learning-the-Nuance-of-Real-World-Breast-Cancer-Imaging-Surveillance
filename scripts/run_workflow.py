#!/usr/bin/env python
"""Run the complete licensed-data reproduction workflow from a YAML file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


REQUIRED_RAW_FILES = [
    "patient.parquet",
    "biomarker.parquet",
    "care_goal.parquet",
    "condition.parquet",
    "imaging.parquet",
    "staging.parquet",
    "surgery.parquet",
    "tumor_exam.parquet",
]
MODEL_SCRIPTS = {
    "linear": "linear_multitask.py",
    "xgboost": "xgboost_multitask.py",
    "mlp": "mlp_multitask.py",
    "gru": "gru_multitask.py",
    "lstm": "lstm_multitask.py",
    "decoder_transformer": "decoder_transformer_multitask.py",
}
VALID_FEATURE_SETS = {"DS", "DSI", "DSM", "DSB", "DSIM", "DSIB", "DSMB", "DSIMB"}


def resolve_path(repo_root: Path, value: str, name: str, required: bool = True) -> Path | None:
    if value is None or not str(value).strip():
        if required:
            raise ValueError(f"Set paths.{name} in the workflow YAML")
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def run_step(label: str, command: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n{'=' * 80}\n{label}\n{printable}\n{'=' * 80}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=env["REPRO_REPO_ROOT"], env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without running them")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8")) or {}
    if config.get("version") != 1:
        raise ValueError("workflow YAML must contain version: 1")

    path_config = config.get("paths", {})
    stages = config.get("stages", {})
    needs_raw_data = stages.get("extract_raw_data", True) or stages.get("prepare_sequence_data", True)
    raw_data_dir = resolve_path(
        repo_root, path_config.get("raw_data_dir"), "raw_data_dir", required=needs_raw_data
    )
    raw_extract_dir = resolve_path(repo_root, path_config.get("raw_extract_dir"), "raw_extract_dir")
    sequence_data_dir = resolve_path(repo_root, path_config.get("sequence_data_dir"), "sequence_data_dir")
    tabular_data_dir = resolve_path(repo_root, path_config.get("tabular_data_dir"), "tabular_data_dir")
    baseline_dir = resolve_path(repo_root, path_config.get("baseline_dir"), "baseline_dir")
    results_dir = resolve_path(repo_root, path_config.get("model_results_dir"), "model_results_dir")
    aggregate_dir = resolve_path(repo_root, path_config.get("aggregate_results_dir"), "aggregate_results_dir")
    figures_dir = resolve_path(repo_root, path_config.get("figures_dir"), "figures_dir")

    if needs_raw_data:
        if not raw_data_dir.is_dir():
            raise FileNotFoundError(f"Licensed raw-data directory not found: {raw_data_dir}")
        missing = [name for name in REQUIRED_RAW_FILES if not (raw_data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing required ConcertAI files: " + ", ".join(missing))
    training = config.get("training", {})
    models = [str(value) for value in training.get("models", MODEL_SCRIPTS)]
    unknown_models = sorted(set(models) - set(MODEL_SCRIPTS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}; valid models: {sorted(MODEL_SCRIPTS)}")
    feature_sets = [str(value).upper() for value in training.get("feature_sets", sorted(VALID_FEATURE_SETS))]
    unknown_features = sorted(set(feature_sets) - VALID_FEATURE_SETS)
    if unknown_features:
        raise ValueError(f"Unknown feature sets: {unknown_features}; valid sets: {sorted(VALID_FEATURE_SETS)}")

    env = os.environ.copy()
    env.update({
        "REPRO_REPO_ROOT": str(repo_root),
        "RAW_EXTRACT_DIR": str(raw_extract_dir),
        "MULTITASK_DATA_DIR": str(sequence_data_dir),
        "MULTITASK_TABULAR_DATA_DIR": str(tabular_data_dir),
        "MODEL_RESULTS_DIR": str(results_dir),
        "AGGREGATE_OUTPUT_DIR": str(aggregate_dir),
        "HYPERBAND_INITIAL_CONFIGS": str(int(training.get("hyperband_initial_configs", 36))),
        "MODEL_EPOCHS": str(int(training.get("sequence_epochs", 20))),
        "MLP_FINAL_EPOCHS": str(int(training.get("mlp_final_epochs", 20))),
        "MLP_TRIAL_EPOCHS": str(int(training.get("mlp_trial_epochs", 20))),
        "MLP_PATIENCE": str(int(training.get("mlp_patience", 5))),
        "XGBOOST_DEVICE": str(training.get("xgboost_device", "auto")),
        "N_BOOTSTRAP": str(int(config.get("aggregation", {}).get("bootstrap_repetitions", 1000))),
        "MPLBACKEND": "Agg",
    })
    if raw_data_dir is not None:
        env["RAW_DATA_DIR"] = str(raw_data_dir)
    tree_method = str(training.get("xgboost_tree_method", "")).strip()
    if tree_method:
        env["XGBOOST_TREE_METHOD"] = tree_method
    else:
        env.pop("XGBOOST_TREE_METHOD", None)

    python = sys.executable
    preprocessing = repo_root / "src" / "preprocessing"
    baselines = repo_root / "src" / "baselines"
    model_dir = repo_root / "src" / "models"
    analysis = repo_root / "src" / "analysis"

    if stages.get("extract_raw_data", True):
        run_step("1/8 Extract raw data", [python, str(preprocessing / "data_extract.py")], env, args.dry_run)
    if stages.get("prepare_sequence_data", True):
        run_step("2/8 Prepare sequence data", [python, str(preprocessing / "prepare_sequence_data.py")], env, args.dry_run)
    if stages.get("prepare_tabular_data", True):
        run_step("3/8 Prepare tabular data", [python, str(preprocessing / "prepare_tabular_data.py")], env, args.dry_run)

    if stages.get("run_baselines", True):
        run_step(
            "4/8 Guideline baseline",
            [python, str(baselines / "guideline_interval_baseline.py"), "--data-dir", str(sequence_data_dir),
             "--output-dir", str(baseline_dir / "guideline_interval")],
            env,
            args.dry_run,
        )
        run_step(
            "4/8 Average same-type baseline",
            [python, str(baselines / "average_same_type_interval_baseline.py"), "--data-dir", str(sequence_data_dir),
             "--output-dir", str(baseline_dir / "average_same_type_interval")],
            env,
            args.dry_run,
        )

    if stages.get("train_models", True):
        for model in models:
            for feature_set in feature_sets:
                model_env = env.copy()
                model_env["EXPERIMENT_NAME"] = feature_set
                run_step(
                    f"5/8 Train {model} with {feature_set}",
                    [python, str(model_dir / MODEL_SCRIPTS[model])],
                    model_env,
                    args.dry_run,
                )

    if stages.get("match_baselines", True):
        run_step(
            "6/8 Build matched baseline inputs",
            [
                python,
                str(analysis / "build_matched_baseline_inputs.py"),
                "--data-dir", str(sequence_data_dir),
                "--results-dir", str(results_dir),
                "--average-baseline-dir", str(baseline_dir / "average_same_type_interval"),
            ],
            env,
            args.dry_run,
        )

    if stages.get("aggregate_results", True):
        run_step(
            "7/8 Generate aggregate results",
            [python, str(analysis / "compute_metrics_and_tables.py")],
            env,
            args.dry_run,
        )

    if stages.get("generate_figures", True):
        run_step(
            "8/8 Generate baseline figure",
            [python, str(analysis / "plot_baseline_comparisons.py"), "--aggregate-dir", str(aggregate_dir),
             "--output-dir", str(figures_dir)],
            env,
            args.dry_run,
        )
        run_step(
            "8/8 Generate feature-contribution figure",
            [python, str(analysis / "plot_feature_contributions.py"), "--input",
             str(aggregate_dir / "feature_group_contributions.csv"), "--output-dir", str(figures_dir)],
            env,
            args.dry_run,
        )

    print("\nWorkflow completed successfully." if not args.dry_run else "\nDry run completed successfully.")


if __name__ == "__main__":
    main()
