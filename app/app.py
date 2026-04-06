"""Road Damage AI — Flask Application (Multi-Model Ensemble, Multi-Page)."""

import base64
import io
from pathlib import Path
from uuid import uuid4

import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from utils.detector import get_model_info, run_inference, run_inference_on_array


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["RESULT_FOLDER"] = RESULT_FOLDER
    app.config["SECRET_KEY"] = "rda-secret-key-change-in-prod"

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    # ── Pages ────────────────────────────────────────────────────────────

    @app.route("/")
    def landing():
        """Hero landing page."""
        return render_template("landing.html")

    @app.route("/dashboard")
    def dashboard():
        """Mode selector — Live Camera or Image Upload."""
        models = get_model_info()
        return render_template("dashboard.html", models=models)

    @app.route("/upload")
    def upload_page():
        """Image upload page."""
        models = get_model_info()
        return render_template("index.html", models=models)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Handle image upload and run ensemble detection."""
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for("upload_page"))

        file = request.files["file"]

        if file.filename == "":
            flash("Please choose an image to upload.", "error")
            return redirect(url_for("upload_page"))

        if not _allowed_file(file.filename):
            flash("Invalid file type. Please upload a JPG or PNG image.", "error")
            return redirect(url_for("upload_page"))

        filename = _uniquify_filename(file.filename)
        upload_path = app.config["UPLOAD_FOLDER"] / filename
        result_filename = f"{Path(filename).stem}_result{Path(filename).suffix}"
        result_path = app.config["RESULT_FOLDER"] / result_filename

        file.save(upload_path)

        try:
            result_data = run_inference(upload_path, result_path)
        except Exception as exc:
            flash(f"Detection failed: {exc}", "error")
            upload_path.unlink(missing_ok=True)
            return redirect(url_for("upload_page"))

        models = get_model_info()
        return render_template(
            "result.html",
            original_image=url_for("static", filename=f"uploads/{filename}"),
            result_image=url_for("static", filename=f"results/{result_filename}"),
            detections=result_data["detections"],
            count=result_data["count"],
            model_counts=result_data["model_counts"],
            consensus_detections=result_data["consensus_detections"],
            models_loaded=result_data["models_loaded"],
            models=models,
        )

    # ── Live Camera API ──────────────────────────────────────────────────

    @app.route("/api/live-frame", methods=["POST"])
    def live_frame():
        """
        Receive a base64-encoded JPEG frame from the browser webcam,
        run the ensemble pipeline, and return detection JSON.
        """
        data = request.get_json(silent=True)
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 JPEG → numpy BGR array
        try:
            header, encoded = data["frame"].split(",", 1)
            img_bytes = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame_array = np.array(pil_img)[:, :, ::-1].copy()  # RGB→BGR
        except Exception as exc:
            return jsonify({"error": f"Frame decode failed: {exc}"}), 400

        # Run inference on the numpy array directly (no file I/O)
        try:
            result = run_inference_on_array(frame_array)
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {exc}"}), 500

        return jsonify(result)

    # ── JSON APIs ────────────────────────────────────────────────────────

    @app.route("/api/detect", methods=["POST"])
    def api_detect():
        """Programmatic JSON detection endpoint for uploaded files."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "" or not _allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400

        filename = _uniquify_filename(file.filename)
        upload_path = app.config["UPLOAD_FOLDER"] / filename
        result_filename = f"{Path(filename).stem}_result{Path(filename).suffix}"
        result_path = app.config["RESULT_FOLDER"] / result_filename

        file.save(upload_path)

        try:
            result_data = run_inference(upload_path, result_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        result_data["result_image"] = url_for(
            "static", filename=f"results/{result_filename}", _external=True
        )
        return jsonify(result_data)

    @app.route("/api/models")
    def api_models():
        return jsonify(get_model_info())

    return app


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _uniquify_filename(filename: str) -> str:
    stem = Path(secure_filename(filename)).stem
    ext = Path(filename).suffix.lower()
    return f"{stem}_{uuid4().hex[:8]}{ext}"


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
