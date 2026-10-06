# Beyond Rigid Guidelines: Learning the Nuance of Real-World Breast Cancer Imaging Surveillance

This repository accompanies the manuscript **Beyond Rigid Guidelines: Learning the Nuance of Real-World Breast Cancer Imaging Surveillance**. It contains the preprocessing, baseline, model-training, evaluation, aggregation, and plotting code used for the manuscript, together with non-patient-level aggregate results.

The data source is **ConcertAI RWD360**, a licensed real-world oncology database. The underlying clinical data and derived patient-level datasets cannot be shared under the data license. This repository provides model-implementation and analysis scripts, input schemas, selected hyperparameters, and manuscript-level aggregate results. Users must obtain their own authorized data access to run preprocessing, training, and patient-level evaluation. See [Data availability](DATA_AVAILABILITY.md).

## Authors

Hanqi Chen, Anthony Girard, Arsh Shah, Rou-Zhen Chen, Winnie Lau, Ravi B. Parikh, Paramita Chatterjee, and Shengpu Tang.

**Affiliation:** Emory University, Atlanta, GA, USA.

## Ethics and IRB

The study used de-identified clinical records from ConcertAI RWD360. **IRB protocol number: [IRB NUMBER TO BE ADDED].**

## Run the analysis notebook without clinical data

Use Python 3.10 and run these commands from this repository's root:

```bash
python -m pip install -r requirements-analysis.txt
python -m jupyterlab notebooks/analysis.ipynb
```

Select **Run All Cells**. [The analysis notebook](notebooks/analysis.ipynb) reads only the included aggregate CSVs, displays Table II, the selected-model subgroup tables and per-scan-type results, and generates Figures 2 and 3 in `generated_figures/`. It does not require patient data, a GPU, or trained models.

For non-interactive execution:

```bash
python -m nbconvert --to notebook --execute notebooks/analysis.ipynb --output analysis.executed.ipynb --output-dir generated_figures --ExecutePreprocessor.timeout=300
```

This reproduces presentation of the saved aggregate results. Recomputing metrics, confidence intervals, preprocessing, or training requires the underlying licensed data and restricted model outputs. The original research-workspace notebooks are not part of this public workflow.

## Repository contents

```text
notebooks/           Runnable aggregate-only analysis notebook
aggregate_results/   Aggregate metrics used in the manuscript
configs/             Selected hyperparameters for six models and eight feature sets
data_schema/         Expected restricted-data interfaces (schema only)
src/analysis/        Matching, aggregation, evaluation, and plotting code
src/baselines/       Guideline and average same-type interval baselines
src/models/          Linear, XGBoost, MLP, GRU, LSTM, and decoder-only models
src/preprocessing/   Raw extraction and sequence/tabular data preparation
```

## Manuscript figures and tables

| Manuscript item | Repository source | Data needed |
| --- | --- | --- |
| Figure 1: multitask framework overview | Conceptual diagram in the manuscript; no figure-generation script is provided | None |
| Table I: hyperparameter search spaces | Search-space definitions in `src/models/`; selected study configurations in `configs/selected_hyperparameters.json` | None to inspect; licensed inputs to rerun search |
| Table II: model performance across feature sets | `aggregate_results/table_ii_metrics.csv`; displayed in `notebooks/analysis.ipynb` | Included aggregates to display; restricted predictions to recompute |
| Table III: XGBoost-DSI subgroup performance | XGBoost-DSI rows in `aggregate_results/table_iii_subgroups.csv`; displayed in the notebook | Included aggregates to display; restricted predictions and staging inputs to recompute |
| Figure 2: matched interval baseline comparisons | `src/analysis/plot_baseline_comparisons.py`; `guideline_comparison.csv` and `average_comparison.csv` | Included aggregates |
| Figure 3: feature-group ablation contributions | `src/analysis/plot_feature_contributions.py`; `feature_group_contributions.csv` | Included aggregates |

The notebook also displays the saved GRU-DSIMB subgroup results, per-scan-type precision/recall/F1, and additional metric and confidence-interval columns. These provide supporting analysis beyond the columns shown in the manuscript tables.

## Full model-workflow environment

Python 3.10 was used for the frozen analysis.

```bash
python -m pip install -r requirements.txt
```

CUDA is optional. XGBoost and PyTorch fall back to CPU when a compatible accelerator is unavailable.

## One-command workflow

Copy the committed example configuration to the ignored local configuration:

```bash
cp configs/workflow.example.yaml configs/workflow.local.yaml
```

PowerShell equivalent:

```powershell
Copy-Item configs/workflow.example.yaml configs/workflow.local.yaml
```

Edit `configs/workflow.local.yaml` and set the directory containing the authorized raw Parquet files:

```yaml
paths:
  raw_data_dir: '<PATH_TO_AUTHORIZED_RAW_PARQUET_DIRECTORY>'
```

The local configuration is ignored by Git because it may contain a private path. Keep this local configuration out of the public release.

Training and execution options are configured in the same file. These include the pipeline stages to run, six model types, eight feature sets, initial Hyperband candidate count, sequence and MLP epochs, MLP patience, XGBoost device settings, and bootstrap repetitions. The committed defaults reproduce the study settings.

Start the entire workflow with:

```bash
bash scripts/run_full_workflow.sh
```

On Windows Command Prompt or PowerShell, run the Python orchestrator directly:

```powershell
python scripts/run_workflow.py --config configs/workflow.local.yaml
```

To use a particular Python environment:

```bash
PYTHON_BIN=/path/to/environment/bin/python bash scripts/run_full_workflow.sh
```

Check the configured raw-file presence and print planned commands without executing the stages (this does not validate data contents or model results):

```bash
bash scripts/run_full_workflow.sh configs/workflow.local.yaml --dry-run
```

Windows equivalent:

```powershell
python scripts/run_workflow.py --config configs/workflow.local.yaml --dry-run
```

The shell script runs each selected model/feature-set experiment sequentially. On a computing cluster, submit this shell script through the local scheduler with the required CPU, memory, GPU, and wall-time allocation. Scheduler directives are intentionally not included because they are system-specific.

## Full reproduction with licensed ConcertAI data

Run all commands from the repository root. Everything under `data/` and `restricted_artifacts/` is ignored by Git and must remain restricted.

### 1. Place the licensed source files

Create `data/raw/` and place the authorized ConcertAI Parquet tables there:

```text
data/raw/
├── patient.parquet
├── biomarker.parquet
├── care_goal.parquet
├── condition.parquet
├── imaging.parquet
├── staging.parquet
├── surgery.parquet
└── tumor_exam.parquet
```

If the files are stored elsewhere, set `RAW_DATA_DIR` to the directory containing the raw Parquet files. For example, in PowerShell:

```powershell
$env:RAW_DATA_DIR = "<PATH_TO_AUTHORIZED_RAW_PARQUET_DIRECTORY>"
```

### 2. Extract the study cohort and source features

```bash
python src/preprocessing/data_extract.py
```

By default, this creates restricted intermediate files in:

```text
data/raw_extract/          Extracted demographic, imaging, staging, surgery, and biomarker tables
data/scan_sequence/        Imaging trajectories
data/raw_data.sqlite       Restricted SQLite copy of the extracted tables
```

The extraction applies the cohort and imaging-history definitions used in the study. `RAW_EXTRACT_DIR` may be set to override `data/raw_extract/`.

### 3. Prepare sequence and tabular model inputs

```bash
python src/preprocessing/prepare_sequence_data.py
python src/preprocessing/prepare_tabular_data.py
```

The first command writes the five sequence-model inputs to `data/prepared/multitask/`:

```text
demographic.csv
imaging_timeseries.csv
staging_timeseries.csv
mastectomy_timeseries.csv
biomarker_timeseries.csv
```

The second writes the tabular-model inputs to `data/prepared/multitask_tabular/`:

```text
target.csv
demographic.csv
imaging_history.csv
staging_history.csv
mastectomy_history.csv
biomarker_history.csv
metadata.json
```

The output locations can be overridden with `MULTITASK_DATA_DIR` and `MULTITASK_TABULAR_DATA_DIR`.

### 4. Run the non-learning baselines

```bash
python src/baselines/guideline_interval_baseline.py --data-dir data/prepared/multitask --output-dir restricted_artifacts/baselines/guideline_interval
python src/baselines/average_same_type_interval_baseline.py --data-dir data/prepared/multitask --output-dir restricted_artifacts/baselines/average_same_type_interval
```

These directories contain patient-level baseline predictions and metrics and must not be committed.

### 5. Train the six multitask models

The eight feature sets are `DS`, `DSI`, `DSM`, `DSB`, `DSIM`, `DSIB`, `DSMB`, and `DSIMB`. If `EXPERIMENT_NAME` is omitted, each model script runs all eight sets serially. To run only one set, set it before launching the script:

```powershell
$env:EXPERIMENT_NAME = "DSI"
python src/models/xgboost_multitask.py
Remove-Item Env:EXPERIMENT_NAME
```

Run all eight feature sets for each model:

```bash
python src/models/linear_multitask.py
python src/models/xgboost_multitask.py
python src/models/mlp_multitask.py
python src/models/gru_multitask.py
python src/models/lstm_multitask.py
python src/models/decoder_transformer_multitask.py
```

Tabular models read `data/prepared/multitask_tabular/`; sequence models read `data/prepared/multitask/`. Each run performs the normalized Hyperband search and writes its report, selected configuration, metrics, prediction details, and training artifacts to a timestamped directory under:

```text
restricted_artifacts/results/
```

Use `MODEL_RESULTS_DIR` to override this result location. The paper configurations are also recorded in `configs/selected_hyperparameters.json`.

### 6. Match the GRU-DSIMB predictions with the two baselines

After the model and average-baseline runs finish, create the restricted matched inputs used by the manuscript comparison:

```bash
python src/analysis/build_matched_baseline_inputs.py
```

This selects the most recent GRU-DSIMB run and writes patient-level matched files to:

```text
restricted_artifacts/results/guideline_matched_comparison/
restricted_artifacts/results/average_matched_comparison/
```

### 7. Generate the shareable aggregate CSVs

```bash
python src/analysis/compute_metrics_and_tables.py
```

The script reads all model prediction outputs and matched baseline files, recomputes the manuscript metrics and patient-bootstrap confidence intervals, and writes the following non-patient-level files to `aggregate_results/`:

```text
table_ii_metrics.csv
table_iii_subgroups.csv
xgboost_dsi_per_scan_type_f1.csv
guideline_comparison.csv
average_comparison.csv
feature_group_contributions.csv
```

The first run performs 1,000 patient-bootstrap repetitions and can take substantial time. Its ignored cache allows interrupted or later runs to continue. If custom restricted locations are used, set `MODEL_RESULTS_DIR`, `MULTITASK_DATA_DIR`, and `MULTITASK_TABULAR_DATA_DIR` before running the script.

This aggregation script targets the original study cohort: it checks for 28,830 held-out events, 4,713 held-out patients, and 330 ultrasound-to-ultrasound comparisons, and expects all 48 model/feature-set results. It is not a general evaluator for arbitrary replacement datasets or partial training runs. Users adapting the model implementations to a different cohort must adapt and validate the study-specific aggregation and subgroup assumptions. Do not disable these checks merely to claim reproduction of the manuscript.

### 8. Reproduce the manuscript plots

```bash
python src/analysis/plot_baseline_comparisons.py
python src/analysis/plot_feature_contributions.py
```

These two plotting scripts require only the shareable files in `aggregate_results/` and write the manuscript-style PDFs to `generated_figures/`.

## Aggregate-only reproduction

Users without ConcertAI access can still reproduce Figures 2 and 3 directly from the included aggregate CSVs:

```bash
python src/analysis/plot_baseline_comparisons.py
python src/analysis/plot_feature_contributions.py
```

## Data and privacy

See `DATA_AVAILABILITY.md`. The `.gitignore` is deliberately defensive, but it is not a substitute for manually reviewing every staged file. Never commit `data/`, `restricted_artifacts/`, prediction-detail tables, incorrect-prediction tables, checkpoints, bootstrap caches, dates, or patient identifiers.

## Reproducibility and license

The code is provided under the [MIT License](LICENSE). That software license grants no rights to the underlying ConcertAI data. The study's selected configurations are recorded in `configs/selected_hyperparameters.json`; the training scripts rerun hyperparameter search, so results may vary with data versions, package versions, and hardware. The saved aggregate CSVs preserve the reported study results.
