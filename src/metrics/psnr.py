from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)."""
    if maxval is None:
        maxval = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)
