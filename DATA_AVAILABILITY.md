# Data Availability

The clinical records used in this study come from the proprietary, licensed ConcertAI RWD360 real-world oncology database. The license and institutional data-use controls do not permit redistribution of the raw records or derived patient- and event-level datasets.

Accordingly, this repository does not contain raw Parquet files, prepared training CSVs, patient identifiers, dates, prediction rows, matched-event rows, incorrect-prediction examples, model checkpoints trained on restricted records, or bootstrap resamples/caches derived from row-level predictions.

The repository provides:

- complete code for extraction, preprocessing, modeling, baselines, evaluation, and aggregate plotting;
- schema-level descriptions of the required restricted inputs;
- selected hyperparameters;
- manuscript-level aggregate metrics, counts, confidence intervals, and feature contributions; and
- an aggregate-only analysis notebook that runs without access to clinical records.

Researchers seeking access to the underlying data must obtain it from the data provider under an appropriate license and complete any required institutional review and security processes. The authors cannot transfer or sublicense these records. Users must obtain their own authorized data access; cloning this repository does not provide data access. The MIT software license does not apply to or sublicense the clinical data.

The aggregate-only notebook displays saved results and recreates figures. It does not recompute patient-level metrics or bootstrap intervals. See the README for the ethics and IRB information.
