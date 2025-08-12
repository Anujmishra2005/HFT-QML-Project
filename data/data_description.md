# Data Description

Place raw HFT CSV files in `data/raw/`.

## Required columns
- Datetime (ISO format or pandas-parsable) — should be the first column and will be used as index.
- Open
- High
- Low
- Close
- Volume

## Notes
- Filenames placed in `data/raw/` should be named clearly, e.g., `AAPL_1m.csv`, `NIFTY_1m.csv`.
- The preprocessing script `src.data_preprocessing` will read CSVs from `data/raw/`, compute technical indicators, create labels, and save processed CSVs into `data/processed/` with the naming convention: `<original_stem>_processed.csv`.
- Ensure timestamps are sorted ascending and there are no duplicate indices.
