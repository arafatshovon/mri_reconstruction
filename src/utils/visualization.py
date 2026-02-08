from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction_triplet(
    zero_filled: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    index: int,
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(zero_filled[index], cmap="gray")
    plt.title("Zero-Fill Reconstruction")

    plt.subplot(1, 3, 2)
    plt.imshow(target[index], cmap="gray")
    plt.title("Ground-Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction[index], cmap="gray")
    plt.title("Prediction")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_error_map(
    target: np.ndarray, prediction: np.ndarray, index: int, save_path: Optional[str] = None
):
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(np.abs(target[index] - prediction[index]), cmap="viridis")
    plt.colorbar()
    plt.title("Error Map")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
