from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from flask_cors import CORS

from Backend.src.pipeline import run_analysis

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Deepfake detection API is running"})


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files or "image" not in request.files:
        return jsonify({"error": "Both video and image files are required"}), 400

    video_file = request.files["video"]
    image_file = request.files["image"]

    if not video_file or not image_file:
        return jsonify({"error": "Missing uploaded files"}), 400

    if video_file.filename == "" or image_file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported video format"}), 400

    if not allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Unsupported image format"}), 400

    video_name = secure_filename(video_file.filename)
    image_name = secure_filename(image_file.filename)

    video_path = UPLOAD_DIR / video_name
    image_path = UPLOAD_DIR / image_name
    plot_filename = "analysis_plot.png"
    plot_path = OUTPUT_DIR / plot_filename

    video_file.save(video_path)
    image_file.save(image_path)

    try:
        result = run_analysis(video_path, image_path, plot_path)

        return jsonify({
            "message": "Analysis completed",
            "prediction": result["prediction"]["label"],
            "score": result["prediction"]["score"],
            "threshold": result["prediction"]["threshold"],
            "processed_frames": result["processed_frames"],
            "detected_frames": result["detected_frames"],
            "plot_url": f"/outputs/{plot_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)