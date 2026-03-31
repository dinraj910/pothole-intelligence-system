from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from utils.detector import run_inference


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def create_app() -> Flask:
	app = Flask(__name__, static_folder="static", template_folder="templates")
	app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
	app.config["RESULT_FOLDER"] = RESULT_FOLDER
	app.config["SECRET_KEY"] = "dev-secret-key-change-me"

	UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
	RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

	@app.route("/")
	def index():
		return render_template("index.html")

	@app.route("/predict", methods=["POST"])
	def predict():
		if "file" not in request.files:
			flash("No file part in the request.", "error")
			return redirect(url_for("index"))

		file = request.files["file"]

		if file.filename == "":
			flash("Please choose an image to upload.", "error")
			return redirect(url_for("index"))

		if not _allowed_file(file.filename):
			flash("Invalid file type. Please upload a JPG or PNG image.", "error")
			return redirect(url_for("index"))

		filename = _uniquify_filename(file.filename)
		upload_path = app.config["UPLOAD_FOLDER"] / filename
		result_filename = f"{Path(filename).stem}_result{Path(filename).suffix}"
		result_path = app.config["RESULT_FOLDER"] / result_filename

		file.save(upload_path)

		try:
			run_inference(upload_path, result_path, conf=0.05)
		except Exception as exc:  # pragma: no cover - safeguard for runtime issues
			flash(f"Detection failed: {exc}", "error")
			if upload_path.exists():
				upload_path.unlink(missing_ok=True)
			return redirect(url_for("index"))

		return render_template(
			"result.html",
			original_image=url_for("static", filename=f"uploads/{filename}"),
			result_image=url_for("static", filename=f"results/{result_filename}"),
		)

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
