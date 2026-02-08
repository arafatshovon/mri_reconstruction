from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)."""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() - gt.min() if maxval is None else maxval

    ssim_total = 0.0
    for slice_num in range(gt.shape[0]):
        ssim_total = ssim_total + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim_total / gt.shape[0]
