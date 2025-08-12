# Data Description

This folder contains the datasets for High-Frequency Trading (HFT) project.

## Folder Structure

- `raw/`  
  Contains raw input datasets in CSV format. Each CSV should include:
  - Datetime (ISO format or pandas-parsable) as the index or first column
  - Open, High, Low, Close prices
  - Volume

- `processed/`  
  Contains cleaned and feature-engineered datasets. These datasets are derived from the raw data by applying preprocessing steps such as:
  - Handling missing values
  - Calculating technical indicators (e.g., MACD, RSI, Bollinger Bands)
  - Label generation (e.g., buy/sell/hold signals)
  - Normalization/scaling

## CSV Requirements

Each raw CSV file should have the following columns:

| Column Name | Description                 |
|-------------|-----------------------------|
| Datetime    | Timestamp of the record     |
| Open        | Opening price of the period |
| High        | Highest price               |
| Low         | Lowest price                |
| Close       | Closing price              |
| Volume      | Volume traded              |

The processed CSV files will have additional columns for technical indicators and targets.

## Notes

- Ensure that timestamps are sorted in ascending order.
- Processed datasets should be saved in the `processed/` folder with the naming convention `<original_filename>_processed.csv`.
- The preprocessing scripts are located in the `src/` directory.

