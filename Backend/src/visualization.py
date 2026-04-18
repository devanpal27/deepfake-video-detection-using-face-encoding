from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.metrics import smooth_data


def save_plot(
    similarities_to_real,
    similarities_to_prev,
    output_path,
    threshold=0.035,
    window_size=5
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot raw similarity to real
    if similarities_to_real:
        x_real = np.arange(len(similarities_to_real))
        plt.plot(x_real, similarities_to_real, alpha=0.35, label="Raw Similarity to Real Image")

        smoothed_real = smooth_data(similarities_to_real, window_size=window_size)
        if smoothed_real:
            x_real_smooth = np.arange(len(smoothed_real)) + (window_size - 1)
            plt.plot(x_real_smooth, smoothed_real, linewidth=2, label="Smoothed Similarity to Real Image")

    # Plot raw similarity to previous
    if similarities_to_prev:
        x_prev = np.arange(len(similarities_to_prev))
        plt.plot(x_prev, similarities_to_prev, alpha=0.35, label="Raw Similarity to Previous Frame")

        smoothed_prev = smooth_data(similarities_to_prev, window_size=window_size)
        if smoothed_prev:
            x_prev_smooth = np.arange(len(smoothed_prev)) + (window_size - 1)
            plt.plot(x_prev_smooth, smoothed_prev, linewidth=2, label="Smoothed Similarity to Previous Frame")

    # Threshold line
    plt.axhline(y=threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")

    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.title("Deepfake Detection Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
