<div align="center">

![Header](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:1f7aec,100:0a1931&text=Road%20Damage%20AI&fontColor=ffffff&fontSize=42&fontAlignY=35&desc=Multi-Model%20YOLOv8%20Ensemble%20%7C%20Live%20Camera%20%26%20Image%20Analysis&descAlignY=56)

[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=github)](#) 
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge&logo=unlicense)](#license) 
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](#-deployment) 

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](#tech-stack)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask)](#tech-stack)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-1f7aec?style=for-the-badge&logo=pythons)](#tech-stack)
[![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](#tech-stack)

<img src="https://readme-typing-svg.demolab.com?font=Manrope&weight=700&size=24&pause=1200&color=1F7AEC&center=true&vCenter=true&random=false&width=800&lines=🚀+3-Model+Consensus+Detection;📡+Real-time+Webcam+Inference;📊+Live+Damage+Area+Analytics;🎯+Production-ready+Flask+Architecture" alt="Typing SVG" />

</div>

---

## 📖 Overview

**Road Damage AI** is an advanced, multi-model ensemble pipeline built for scalable road quality inspection. It upgrades standard single-model pothole detection by running an ensemble of **three specialized YOLOv8 models** simultaneously. 

By combining models trained on different points of view (Dashcam, Close-up, Street-level) using a cascaded transfer-learning strategy, the application achieves incredible recall and utilizes **Cross-Model Non-Maximum Suppression (NMS)** to track multi-model consensus and filter out false positives.

Users can interface with the AI pipeline via a **Server-Side Rendered Image Upload Dashboard** or a **Live Camera Feed** that runs inference every 2 seconds without requiring client-side GPU power.

---

## ✨ Key Features

| ✅ | Capability |
| --- | --- |
| 🤖 | **3-Model YOLOv8 Ensemble:** RDD2022 (Dashcam), Pothole-600 (Close-up), Kaggle (Street). |
| 🛡️ | **Consensus Tracking:** Overlapping bounding boxes are merged via cross-model NMS (IoU=0.45). Detections flagged by 2+ models earn a high-confidence consensus star. |
| 📡 | **Live Camera Detection:** Uses the `MediaStream API` to capture webcam frames entirely in the browser, passing base64 JPEG blobs to `/api/live-frame` for seamless JS-driven live tracking. |
| 📊 | **Damage Analytics:** Calculates cumulative damage counts and real-time damage area percentages. |
| 🎨 | **Premium Dark-Mode UI:** Glassmorphism accents, live statistics counting, dynamic color-coded bounding boxes (🔴 Model A, 🟢 Model B, 🔵 Model C). |
| 🚀 | **Production-Ready:** Optimized with `gunicorn` (1 worker) so all 3 models load only once into memory, perfectly tuned for Render's free tier. |

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Browser Client] -->|1. Image Upload /predict| B(Flask App Engine)
    A -->|2. Webcam Frame /live-frame| B
    
    subgraph Multi-Model Ensemble [Singleton Inference Engine]
        B --> C{Cross-Model Inference}
        C -->|conf=0.05| M1[Model A: RDD2022 ]
        C -->|conf=0.10| M2[Model B: Pothole-600 ]
        C -->|conf=0.05| M3[Model C: Kaggle ]
        
        M1 --> N[Cross-Model NMS IoU=0.45]
        M2 --> N
        M3 --> N
        N --> D[Consensus Scoring & Data Cleanup]
    end

    D --> E[Render cv2 Bounding Boxes]
    E --> F[Generate JSON & Analytics]
    F -->|Return HTML / JSON| A
```

---

## 📁 Project Structure

```
📦 Pothole-AI-System
├─ app/
│  ├─ app.py                  # Core Flask routing & JSON APIs
│  ├─ RoadDamageAI_Phase1/    
│  │  └─ weights/             # 3 specialized YOLOv8 models
│  │     ├─ model_a_rdd2022.pt
│  │     ├─ model_b_pothole600.pt
│  │     └─ model_c_kaggle.pt
│  ├─ utils/
│  │  └─ detector.py          # Array/Disk inference & NMS logic
│  ├─ static/
│  │  ├─ js/live_cam.js       # Webcam stream & API sync
│  │  ├─ styles.css           # Global Dark Theme UI
│  │  ├─ uploads/             # Ephemeral image uploads
│  │  └─ results/             # Annotated output storage
│  └─ templates/
│     ├─ landing.html         # Hero page
│     ├─ dashboard.html       # Mode Selector (Live vs Upload)
│     ├─ index.html           # Upload Interface
│     └─ result.html          # Detailed tabular analytics
├─ notebook/
│  └─ Road_Damage_MultiModel_Pipeline_final.ipynb # Source training
├─ requirements.txt           # Dependencies
├─ Procfile                   # Gunicorn config for Render
└─ render.yaml                # Render Infrastructure-as-Code
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- `gunicorn` (for Unix environments)

### Local Installation

1. **Clone and Setup Virtual Environment:**
   ```bash
   git clone https://github.com/<your-username>/Pothole-AI-System.git
   cd Pothole-AI-System
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Ensure Weights Are Present:**
   Store your 3 trained YOLOv8 `.pt` models in `app/RoadDamageAI_Phase1/weights/`. 

3. **Run the Development Server:**
   ```bash
   cd app
   python app.py
   # Visit http://127.0.0.1:5000 in your browser
   ```

---

## 🚀 Deployment (Render)

This project includes a `Procfile` and `render.yaml` for one-click deployment on Render.

**Important Note regarding Render's Free Tier:** 
The application restricts `gunicorn` to `--workers 1`. This is done intentionally because keeping three YOLO models in memory consumes ~66MB, and spinning up multiple workers on a 512MB RAM free instance will cause out-of-memory (OOM) crashes.

To deploy:
1. Connect this repo to Render.
2. The `render.yaml` blueprint will automatically detect the settings.
3. Access your live app!

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

## 🗺️ Roadmap (Phase 2 & 3)

- [ ] **Phase 2 — Severity Classification:** Train an additional CNN to classify the detected bounding boxes by severity (Small / Medium / Severe).
- [ ] **Phase 3 — Location Intelligence:** Implement GPS EXIF extraction for image uploads and browser Geolocation API for the live camera to build dynamic pothole maps.
- [ ] **DB Integration:** Migrate to PostgreSQL for maintaining historic detection logs.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push to branch (`git push origin feature/awesome`)
5. Open a Pull Request

---

## 📜 License & Acknowledgments

- Distributed under the MIT License.
- Built using **Ultralytics YOLOv8**.
- Datasets utilized: RDD2022, Pothole-600, Kaggle Pothole Dataset.

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&height=120&color=0:0a1931,100:1f7aec&section=footer)

</div>
