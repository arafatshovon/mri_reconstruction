# Accelerated MRI Reconstruction (VarNet + WCNN)

This repository refactors the completed thesis notebooks into a clean, reproducible research codebase. It targets accelerated MRI reconstruction using a VarNet backbone with a Wavelet CNN (MWCNN) regularizer, trained on fastMRI-style datasets.

## Method Summary
- **Model**: VarNet with cascaded data consistency and a WCNN regularizer.
- **Loss**: L1 + SSIM (fastMRI SSIMLoss), matching the notebooks.
- **Metrics**: PSNR and SSIM computed slice-wise on reconstructed images.

## Dataset
This code expects fastMRI-style HDF5 files and CSV manifests listing `file_name` and `acquisition`.
See `data/README.md` for the expected layout and file naming.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
Knee (default configs):
```bash
scripts/train.sh
```

Brain (M4Raw configs):
```bash
python -m src.train \
  --dataset-config configs/dataset_brain.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train_brain.yaml
```

## Validation
```bash
scripts/validate.sh /path/to/checkpoint.ckpt
```

## Testing
```bash
python -m src.test \
  --dataset-config configs/dataset.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --checkpoint /path/to/checkpoint.ckpt
```

## Reproducing Paper Results
1. Populate `data/raw/` with the same CSV manifests used in the notebooks.
2. Ensure acquisitions and mask settings match the original experiments.
3. Run training with the corresponding configs (knee or brain).
4. Use `src/validate.py` on the best checkpoint to report PSNR/SSIM.

Figures from the thesis are stored in `figures/`. The thesis PDF is available under `paper/`.

## Citation
If you use this repository, please cite the thesis:
```bibtex
@thesis{your_name_2026,
  title     = {Variational Network with Wavelet-based UNET in Accelerated MRI Reconstruction from Under Sampled K-space Data},
  author    = {Yasir Arafat Prodhan, Dr. Shaikh Anowarul Fattah},
  year      = {2026},
  school    = {Bangladesh University of Engineering & Technology}
}
```
