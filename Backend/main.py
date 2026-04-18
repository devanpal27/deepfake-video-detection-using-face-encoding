from pathlib import Path
from Backend.src.pipeline import run_analysis

BASE_DIR = Path(__file__).resolve().parent

video_path = BASE_DIR / "uploads" / "fake.mp4"
image_path = BASE_DIR / "uploads" / "image.png"
plot_path = BASE_DIR / "outputs" / "analysis_plot.png"

result = run_analysis(
    video_path,
    image_path,
    plot_path,
    frame_step=3,
    blur_threshold=80.0,
    duplicate_threshold=2.0
)

print("Prediction:", result["prediction"]["label"])
print("Score:", result["prediction"]["score"])
print("Processed Frames:", result["processed_frames"])
print("Sampled Frames:", result["sampled_frames"])
print("Detected Frames:", result["detected_frames"])
print("Skipped Blurry Frames:", result["skipped_blurry"])
print("Skipped Duplicate Frames:", result["skipped_duplicate"])
print("Plot saved at:", result["plot_path"])