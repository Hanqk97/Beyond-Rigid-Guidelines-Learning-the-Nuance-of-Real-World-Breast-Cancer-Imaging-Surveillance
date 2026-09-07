# Restricted Data Interface

The files described here are interfaces, not distributable sample data. Do not create synthetic rows that could be mistaken for study observations, and do not copy headers plus real records into this repository.

## Sequence-model input

Set `MULTITASK_DATA_DIR` to a restricted directory containing:

- `demographic.csv`: one row per patient.
- `imaging_timeseries.csv`: one row per patient imaging visit.
- `staging_timeseries.csv`: longitudinal stage events.
- `mastectomy_timeseries.csv`: longitudinal surgery events.
- `biomarker_timeseries.csv`: longitudinal biomarker events.

The stable keys and fields are listed in `expected_columns.json`. The scan vector order is `CT, MRI, PET, ULTRASOUND, XRAY, MAMMO_TOMO, NUCLEAR_MED`.

## Tabular-model input

Set `MULTITASK_TABULAR_DATA_DIR` to a restricted directory containing:

- `target.csv`
- `demographic.csv`
- `imaging_history.csv`
- `staging_history.csv`
- `mastectomy_history.csv`
- `biomarker_history.csv`

The target and four history tables join on `patient_id`, `anchor_scan_date`, `anchor_sequence`, and `target_sequence`. The demographic table joins on `patient_id`. History features are flattened into numbered lag columns; `expected_columns.json` documents the naming patterns because the maximum number of lags depends on the licensed cohort.

## Identifier warning

`patient_id`, event dates, sequences, and any joined row containing them are patient/event-level restricted data. The fact that an identifier is encoded, hashed, or vendor-assigned does not make the row suitable for public release.
