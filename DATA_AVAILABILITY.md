# Data Availability

The clinical records used in this study are proprietary, licensed real-world oncology data. The license and institutional data-use controls do not permit redistribution of the raw records or derived patient- and event-level datasets.

Accordingly, this repository does not contain raw Parquet files, prepared training CSVs, patient identifiers, dates, prediction rows, matched-event rows, incorrect-prediction examples, model checkpoints trained on restricted records, or bootstrap resamples/caches derived from row-level predictions.

The repository provides:

- complete code for extraction, preprocessing, modeling, baselines, evaluation, and aggregate plotting;
- schema-level descriptions of the required restricted inputs;
- selected hyperparameters; and
- manuscript-level aggregate metrics, counts, confidence intervals, and feature contributions that do not permit reconstruction of individual records.

Researchers seeking access to the underlying data must obtain it from the data provider under an appropriate license and complete any required institutional review and security processes. The authors cannot transfer or sublicense these records. Additional access/contact details can be added to the camera-ready repository after anonymous review.
