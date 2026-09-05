# Raw data

Expected local dataset: `Flight_Booking.csv`.

The canonical source file supplied for this rebuild contains 300,153 rows and 12 CSV columns, including the export-only `Unnamed: 0` index. Raw CSV files are intentionally excluded from Git by `.gitignore`.

Validate it with:

```bash
python scripts/validate_data.py data/raw/Flight_Booking.csv --strict
```
