import numpy as np


def calculate_similarity(landmarks1, landmarks2, weights=None):
    if landmarks1 is None or landmarks2 is None:
        return None

    if landmarks1.shape != landmarks2.shape:
        return None

    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if len(weights) != len(distances):
            return None
        score = np.average(distances, weights=weights)
    else:
        median_dist = np.median(distances)
        mean_dist = np.mean(distances)
        p90_dist = np.percentile(distances, 90)

        # robust combined score
        score = (0.5 * median_dist) + (0.3 * mean_dist) + (0.2 * p90_dist)

    return float(score)


def smooth_data(data, window_size=5):
    if not data:
        return []

    data = np.asarray(data, dtype=np.float32)

    if len(data) < window_size:
        return data.tolist()

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.convolve(data, kernel, mode="valid")
    return smoothed.tolist()


def classify_video(similarities_to_real, threshold=0.4, window_size=5):
    if not similarities_to_real:
        return {
            "label": "Unable to classify",
            "score": None,
            "threshold": threshold,
            "raw_mean": None,
            "smoothed_mean": None,
            "max_score": None,
            "std_dev": None,
            "anomaly_ratio": None
        }

    raw_scores = np.asarray(similarities_to_real, dtype=np.float32)
    smoothed_scores = np.asarray(smooth_data(similarities_to_real, window_size), dtype=np.float32)

    raw_mean = float(np.mean(raw_scores))
    smoothed_mean = float(np.mean(smoothed_scores)) if len(smoothed_scores) > 0 else raw_mean
    max_score = float(np.max(raw_scores))
    std_dev = float(np.std(raw_scores))
    anomaly_ratio = float(np.mean(raw_scores > threshold))

    # combined decision score
    final_score = (
        0.5 * smoothed_mean +
        0.3 * raw_mean +
        0.2 * max_score
    )

    label = "FAKE" if final_score > threshold else "REAL"

    return {
        "label": label,
        "score": final_score,
        "threshold": threshold,
        "raw_mean": raw_mean,
        "smoothed_mean": smoothed_mean,
        "max_score": max_score,
        "std_dev": std_dev,
        "anomaly_ratio": anomaly_ratio
    }