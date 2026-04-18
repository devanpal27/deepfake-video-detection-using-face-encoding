import React, { useState } from "react";
import "./App.css";

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!videoFile || !imageFile) {
      setError("Please upload both a video and a reference image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("image", imageFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="bg-orb orb-1"></div>
      <div className="bg-orb orb-2"></div>

      <div className="container">
        <div className="hero">
          <p className="tag">AI • Deepfake Detection • Landmark Analysis</p>
          <h1>Deepfake Detection Dashboard</h1>
          <p className="subtitle">
            Upload a video and a real reference image to analyze facial landmark
            consistency and detect manipulation patterns.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="upload-form">
          <div className="upload-grid">
            <label className="file-card">
              <div className="file-card-top">
                <span className="file-icon">🎥</span>
                <div>
                  <h3>Upload Video</h3>
                  <p>Click to select your test video</p>
                </div>
              </div>

              <div className="custom-upload">
                {videoFile ? videoFile.name : "Choose Video File"}
              </div>

              <input
                type="file"
                accept="video/*"
                className="hidden-input"
                onChange={(e) => setVideoFile(e.target.files[0])}
              />
            </label>

            <label className="file-card">
              <div className="file-card-top">
                <span className="file-icon">🖼️</span>
                <div>
                  <h3>Reference Image</h3>
                  <p>Click to select a clear frontal face image</p>
                </div>
              </div>

              <div className="custom-upload">
                {imageFile ? imageFile.name : "Choose Image File"}
              </div>

              <input
                type="file"
                accept="image/*"
                className="hidden-input"
                onChange={(e) => setImageFile(e.target.files[0])}
              />
            </label>
          </div>

          <button type="submit" disabled={loading} className="analyze-btn">
            {loading ? "Processing..." : "Analyze Media"}
          </button>
        </form>

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className="result-card">
            <div className="result-header">
              <h2>Analysis Result</h2>
              <span
                className={`badge ${
                  result.prediction === "FAKE" ? "fake" : "real"
                }`}
              >
                {result.prediction}
              </span>
            </div>

            <div className="stats-grid">
              <div className="stat-box">
                <span className="stat-label">Score</span>
                <span className="stat-value">
                  {result.score !== null && result.score !== undefined
                    ? Number(result.score).toFixed(4)
                    : "N/A"}
                </span>
              </div>

              <div className="stat-box">
                <span className="stat-label">Threshold</span>
                <span className="stat-value">
                  {result.threshold ?? "N/A"}
                </span>
              </div>

              <div className="stat-box">
                <span className="stat-label">Processed Frames</span>
                <span className="stat-value">{result.processed_frames}</span>
              </div>

              <div className="stat-box">
                <span className="stat-label">Detected Frames</span>
                <span className="stat-value">{result.detected_frames}</span>
              </div>
            </div>

            {result.plot_url && (
              <div className="plot-card">
                <h3>Similarity Analysis Graph</h3>
                <img
                  src={result.plot_url}
                  alt="Analysis Plot"
                  className="plot-image"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;