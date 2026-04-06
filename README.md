
<div align="center">

![Header](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:1f7aec,100:0a1931&text=Pothole%20Detection%20System&fontColor=ffffff&fontSize=38&fontAlignY=35&desc=YOLOv8%20%2B%20Flask%20%7C%20Edge-ready&descAlignY=56)

[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=github)](#) 
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge&logo=unlicense)](#license) 
[![PRs](https://img.shields.io/badge/PRs-Welcome-orange?style=for-the-badge&logo=gitbook)](#contributing) 
[![Maintained](https://img.shields.io/badge/Maintained-Yes-blue?style=for-the-badge&logo=dependabot)](#) 

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#tech-stack)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)](#tech-stack)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-1f7aec?style=for-the-badge&logo=pythons)](#tech-stack)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](#tech-stack)

<img src="https://readme-typing-svg.demolab.com?font=Manrope&weight=700&size=24&pause=1200&color=1F7AEC&center=true&vCenter=true&random=false&width=800&lines=🚀+Real-time+pothole+detection+for+safer+roads;⚡+Low-confidence+recall+%28conf=0.05%29+for+edge+scenarios;🎯+Production-ready+Flask+web+app" alt="Typing SVG" />

</div>

---

## 🔗 Quick Navigation

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Deep Dive](#-technical-deep-dive)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Screenshots / Demo](#-screenshots--demo)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)
- [Performance](#-performance-metrics)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)
- [Show Your Support](#-show-your-support)

---

## 📖 Overview

| What | Why |
| --- | --- |
| Web app that detects potholes in road images using a trained YOLOv8 model and surfaces annotated results. | Rapidly surface high-recall pothole detections to support road maintenance and smart mobility dashboards. |

---

## ✨ Features

| ✅ | Capability |
| --- | --- |
| 🚀 | YOLOv8 inference with low confidence (0.05) to maximize recall in noisy real-world captures. |
| ⚡ | Fast Flask API with secure upload handling and caching of model weights. |
| 🖼️ | Automatic overlay of bounding boxes and side-by-side result view. |
| 🛡️ | Strict file validation, flash messaging, and graceful error handling. |
| 📁 | Organized static asset management (uploads/results) for traceability. |
| 📊 | Ready for dashboard integration and further analytics. |

---

## 🏗️ Architecture

```
					+-----------------------+
					|      Browser UI       |
					|  (Upload & Results)   |
					+----------+------------+
										 |
										 v
					+----------+------------+
					|      Flask App        |
					|  Routes: / , /predict |
					+----------+------------+
										 |
					 Secure upload & validate
										 |
										 v
					+----------+------------+
					|   YOLOv8 Inference    |
					|  (Ultralytics, conf)  |
					+----------+------------+
										 |
										 v
					+----------+------------+
					|  Result Overlay (cv2) |
					+----------+------------+
										 |
										 v
					+----------+------------+
					|  Static Assets Store  |
					|  uploads/ & results/  |
					+-----------------------+
```

---

## 🔍 Technical Deep Dive

<details>
<summary>Click to expand</summary>

- **Model Loading**: Singleton pattern via `functools.lru_cache` keeps YOLO weights hot-loaded.
- **Inference Path**: Upload → secure filename → save → YOLO forward pass with `conf=0.05` → OpenCV render to results directory.
- **Error Handling**: Flash messaging for missing/invalid files; guarded exceptions around model calls; temp cleanup on failure.
- **Security**: Extension whitelist, `secure_filename`, isolated static folders, predictable paths.
- **UX**: Drag-and-drop upload, loader overlay during inference, immediate result view.
- **Extensibility**: Swap weights by replacing `app/model/best.pt`; adjust confidence/IoU in `detector.py`.

</details>

---

## 📁 Project Structure

```
📦 Pothole-AI-System
├─ app/
│  ├─ app.py
│  ├─ model/                  # YOLO weights (best.pt)
│  ├─ static/
│  │  ├─ styles.css
│  │  ├─ uploads/             # user uploads
│  │  └─ results/             # annotated outputs
│  ├─ templates/
│  │  ├─ index.html
│  │  └─ result.html
│  └─ utils/
│     └─ detector.py          # model loader + inference
├─ notebook/                  # training/experiments
│  └─ Pothole_Detection.ipynb
├─ requirements.txt
└─ README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip
- GPU optional (CPU works for demo; GPU recommended for speed)

### Installation

```bash
git clone https://github.com/<your-username>/Pothole-AI-System.git
cd Pothole-AI-System
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ensure the model is placed
mkdir -p app/model
cp app/models/best.pt app/model/best.pt  # or your own weights

python app/app.py
# Visit http://localhost:5000
```

---

## 🖼️ Screenshots / Demo

![alt text](screenshots/1.png)
![alt text](screenshots/2.png)
![alt text](screenshots/3.png)
![alt text](screenshots/4.png)

![alt text](screenshots/5.png)
![alt text](screenshots/6.png)
![alt text](screenshots/7.png)
![alt text](screenshots/8.png)

![alt text](screenshots/9.png)
![alt text](screenshots/10.png)
![alt text](screenshots/11.png)
![alt text](screenshots/12.png)

---

## ⚙️ Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `SECRET_KEY` | Flask session/flash secret | set in `app.py` (change for prod) |
| `UPLOAD_FOLDER` | Upload path | `app/static/uploads` |
| `RESULT_FOLDER` | Annotated outputs | `app/static/results` |
| `MODEL_PATH` | YOLO weights | `app/model/best.pt` |
| `CONFIDENCE` | Detection threshold | `0.05` |

---

## 🛠️ Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](#)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-1F7AEC?style=for-the-badge&logo=ai)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](#)
[![Pillow](https://img.shields.io/badge/Pillow-ffb400?style=for-the-badge&logo=python)](#)

</div>

---

## 📈 Performance Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| Inference latency | ~XX ms (CPU i7) | Update after benchmarking |
| Precision | XX% | From validation run |
| Recall | XX% | Emphasis on recall at conf=0.05 |
| Model size | ~XX MB | YOLOv8 variant used |

> Run your own eval and update the table for transparency.

---

## 🗺️ Roadmap

```mermaid
graph LR
	A[Current: Web Upload] --> B[API Endpoint /predict JSON]
	B --> C[Batch Processing Queue]
	C --> D[Edge Deployment (Jetson)]
	D --> E[Dashboard with Metrics]
	E --> F[Active Learning Loop]
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push to branch (`git push origin feature/awesome`)
5. Open a Pull Request

Please include tests or screenshots where applicable.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 👤 Author

<div align="center">

**Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/your-profile)
[![Portfolio](https://img.shields.io/badge/Portfolio-1f7aec?style=for-the-badge&logo=google-chrome&logoColor=white)](https://your-portfolio.com)
[![Email](https://img.shields.io/badge/Email-Contact-orange?style=for-the-badge&logo=gmail&logoColor=white)](mailto:you@example.com)

</div>

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- OpenCV team
- RDD2022 dataset contributors
- Flask community

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/Pothole-AI-System&type=Date)](https://star-history.com/#your-username/Pothole-AI-System)

---

## 💙 Show Your Support

If you find this project useful:

- ⭐ Star this repo
- 🧠 Share feedback & ideas
- 🐛 File issues for bugs or feature requests

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&height=120&color=0:0a1931,100:1f7aec&section=footer)

</div>

