import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from Backend.src.landmarks import extract_landmarks, normalize_landmarks
from Backend.src.metrics import calculate_similarity, smooth_data, classify_video
from Backend.src.visualization import save_plot

def load_real_landmarks(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    landmarks = extract_landmarks(image)
    landmarks = normalize_landmarks(landmarks)

    if landmarks is None:
        raise ValueError("No face detected in reference image.")

    return landmarks


def is_duplicate_frame(current_frame, previous_frame, duplicate_threshold=2.0):
    if previous_frame is None:
        return False

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(current_gray, previous_gray)
    mean_diff = np.mean(difference)

    return mean_diff < duplicate_threshold


def process_video(
    video_path,
    real_landmarks,
    frame_step=1,
    blur_threshold=20.0,
    duplicate_threshold=0.5
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    similarities_to_real = []
    similarities_to_prev = []
    frame_rates = []

    prev_landmarks = None
    prev_kept_frame = None

    processed_frames = 0
    sampled_frames = 0
    skipped_blurry = 0
    skipped_duplicate = 0
    no_face_frames = 0
    normalize_failed = 0
    detected_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        frame_rates.append(fps)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        sampled_frames += 1


        if is_duplicate_frame(frame, prev_kept_frame, duplicate_threshold=duplicate_threshold):
            skipped_duplicate += 1
            frame_index += 1
            continue

        raw_landmarks = extract_landmarks(frame)
        if raw_landmarks is None:
            no_face_frames += 1

            if no_face_frames <= 5:
                debug_path = f"debug_no_face_{processed_frames}.jpg"
                cv2.imwrite(debug_path, frame)
                print(f"[DEBUG] No face detected on frame {processed_frames}. Saved: {debug_path}")

            frame_index += 1
            continue

        landmarks = normalize_landmarks(raw_landmarks)
        if landmarks is None:
            normalize_failed += 1
            print(f"[DEBUG] Normalize failed on frame {processed_frames}")
            frame_index += 1
            continue

        detected_frames += 1

        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        if sim_to_real is not None:
            similarities_to_real.append(sim_to_real)

        if prev_landmarks is not None:
            sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
            if sim_to_prev is not None:
                similarities_to_prev.append(sim_to_prev)

        prev_landmarks = landmarks
        prev_kept_frame = frame.copy()

        frame_index += 1

    cap.release()

    print("\n===== DEBUG SUMMARY =====")
    print("Processed:", processed_frames)
    print("Sampled:", sampled_frames)
    print("Skipped blurry:", skipped_blurry)
    print("Skipped duplicate:", skipped_duplicate)
    print("No face detected:", no_face_frames)
    print("Normalize failed:", normalize_failed)
    print("Detected frames:", detected_frames)
    print("=========================\n")


    return {
        "processed_frames": processed_frames,
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "skipped_blurry": skipped_blurry,
        "skipped_duplicate": skipped_duplicate,
        "no_face_frames": no_face_frames,
        "normalize_failed": normalize_failed,
        "similarities_to_real": similarities_to_real,
        "similarities_to_prev": similarities_to_prev,
        "frame_rates": frame_rates,
    }




def run_analysis(
    video_path,
    image_path,
    plot_path,
    frame_step=3,
    blur_threshold=80.0,
    duplicate_threshold=2.0
):
    real_landmarks = load_real_landmarks(image_path)

    results = process_video(
        video_path,
        real_landmarks,
        frame_step=frame_step,
        blur_threshold=blur_threshold,
        duplicate_threshold=duplicate_threshold
    )

    prediction = classify_video(
    results["similarities_to_real"],
    threshold=0.4,
    window_size=5
    )

    save_plot(
        results["similarities_to_real"],
        results["similarities_to_prev"],
        plot_path,
        threshold=prediction["threshold"],
        window_size=5
    ) 

    return {
        "prediction": prediction,
        "processed_frames": results["processed_frames"],
        "sampled_frames": results["sampled_frames"],
        "detected_frames": results["detected_frames"],
        "skipped_blurry": results["skipped_blurry"],
        "skipped_duplicate": results["skipped_duplicate"],
        "plot_path": str(plot_path),
    }