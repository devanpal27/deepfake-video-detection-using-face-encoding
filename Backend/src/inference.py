import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
OUTPUT_DIR = BASE_DIR / "outputs"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(MODEL_PATH))


def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return float(np.mean(distances))


def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in shape.parts()])


def moving_average(values, window=15):
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def detect_fake_start(similarities_to_real, similarities_to_prev):
    real_arr = np.array(similarities_to_real, dtype=float)
    prev_arr = np.array(similarities_to_prev, dtype=float)

    real_smooth = moving_average(real_arr, window=15)
    prev_smooth = moving_average(prev_arr, window=15)

    score = real_smooth + (prev_smooth * 2.0)

    start_search = max(10, len(score) // 8)
    end_search = len(score) - 10 if len(score) > 20 else len(score)

    if end_search <= start_search:
        fake_start = len(score) // 2
    else:
        local_idx = np.argmax(score[start_search:end_search])
        fake_start = start_search + int(local_idx)

    return fake_start, real_smooth, prev_smooth


def run_analysis(video_path, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "result": "error",
            "message": "Reference image could not be loaded."
        }

    real_landmarks = get_landmarks(image)
    if real_landmarks is None:
        return {
            "result": "error",
            "message": "No face detected in reference image."
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "result": "error",
            "message": "Video could not be opened."
        }

    similarities_to_real = []
    similarities_to_prev = []
    prev_landmarks = None
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        landmarks = get_landmarks(frame)

        if landmarks is not None:
            sim_real = calculate_similarity(landmarks, real_landmarks)
            similarities_to_real.append(sim_real if sim_real is not None else 0)

            if prev_landmarks is not None:
                sim_prev = calculate_similarity(landmarks, prev_landmarks)
                similarities_to_prev.append(sim_prev if sim_prev is not None else 0)
            else:
                similarities_to_prev.append(0)

            prev_landmarks = landmarks
        else:
            similarities_to_real.append(0)
            similarities_to_prev.append(0)

    cap.release()

    if total_frames == 0:
        return {
            "result": "error",
            "message": "No frames were read from the video."
        }

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_path = OUTPUT_DIR / "analysis_plot.png"

    fake_start_frame, real_smooth, prev_smooth = detect_fake_start(
        similarities_to_real,
        similarities_to_prev
    )

    frames = np.arange(len(similarities_to_real))
    y_max = max(
        max(real_smooth) if len(real_smooth) else 1,
        max(prev_smooth) if len(prev_smooth) else 1,
        1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(frames, real_smooth, label="Similarity to Real Image")
    plt.plot(frames, prev_smooth, label="Variations")
    plt.axvline(x=fake_start_frame, color="red", linestyle="--", label="Deepfake Starts")
    plt.axvspan(fake_start_frame, len(frames) - 1, color="yellow", alpha=0.3, label="Deepfake Region")

    real_x = max(fake_start_frame // 2, 10)
    fake_x = min(fake_start_frame + max(len(frames) // 6, 40), len(frames) - 80)

    plt.text(real_x, y_max * 0.8, "Real", fontsize=15, weight="bold", color="black")
    plt.text(fake_x, y_max * 0.75, "Fake", fontsize=15, weight="bold", color="black")

    plt.annotate(
        "",
        xy=(10, y_max * 0.78),
        xytext=(max(fake_start_frame - 10, 20), y_max * 0.78),
        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2)
    )
    plt.annotate(
        "",
        xy=(fake_start_frame + 20, y_max * 0.78),
        xytext=(len(frames) - 10, y_max * 0.78),
        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2)
    )

    plt.title("Analysis Plot with Detected Deepfake Region")
    plt.xlabel("Frame Index")
    plt.ylabel("Scores / Variations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(plot_path))
    plt.close()

    prediction = "Fake" if fake_start_frame < len(frames) - 1 else "Real"

    return {
        "prediction": prediction,
        "total_frames": total_frames,
        "fake_start_frame": int(fake_start_frame),
        "plot_path": str(plot_path)
    }