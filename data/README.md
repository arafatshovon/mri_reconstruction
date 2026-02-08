# Data Layout

Place raw fastMRI files and CSV manifests in `data/raw/`.

Expected CSV format (from the notebooks):
- `file_name`: path to HDF5 file
- `acquisition`: acquisition name used for filtering

Example files:
- `data/raw/train_knee_multi_coil.csv`
- `data/raw/validation_knee_multi_coil.csv`
- `data/raw/train_M4Raw.csv`
- `data/raw/val_M4Raw.csv`

Processed outputs (if any) can be stored in `data/processed/`.
