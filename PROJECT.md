# Smart Road Damage Detection & Complaint Intelligence System

## 🎯 Project Objective
- End-to-end AI platform to detect potholes using YOLOv8, classify severity (planned CNN), capture user-uploaded road imagery, enrich with geolocation, and generate structured complaints for civic authorities.
- Provide a data-driven pipeline from image/video ingestion to actionable insights for maintenance teams.

## 🧠 Problem Statement
- Road infrastructure monitoring is reactive and manual, leading to delayed maintenance and higher repair costs.
- Existing complaint systems are cumbersome, unstructured, and lack evidence linking (image + location + severity).
- Real-time, geo-tagged reporting is scarce, creating a gap between citizens, field teams, and authorities.

## 💡 Solution Overview
- Automated pothole detection via YOLOv8 with low-confidence tuning for high recall on diverse road scenes.
- Planned severity classification (small/medium/severe) to prioritize repairs and resource allocation.
- Image-to-insight pipeline that links detections with GPS/reverse geocoding for location-aware reports.
- Future complaint automation to generate structured, authority-ready submissions (email/PDF/API when available).

## 🏗️ System Architecture
### Core Components
1. **AI Layer**: YOLOv8 for detection (current); CNN classifier for severity (planned).
2. **Backend**: Flask (current) with a path to FastAPI for scalable APIs.
3. **Frontend**: Web UI (upload, results); mobile/web expansion planned.
4. **Location Intelligence**: GPS capture on client; reverse geocoding (OpenStreetMap/Nominatim); district mapping (Kerala focus initially).
5. **Complaint Engine**: Template-based, structured report generation; future integration with civic APIs.
6. **Database**: Stores uploads, detections, metadata, geo-coordinates, and complaint states.

### Pipeline Flow
```
Image/Video → Detection (YOLOv8) → Severity Classification (CNN, planned) →
Location Enrichment (GPS + reverse geocode) → Structured Complaint Generation →
Dashboards & Reporting
```

## ⚙️ Technologies Used
- Python, Flask (current) / FastAPI (planned)
- YOLOv8 (Ultralytics), OpenCV
- TensorFlow or PyTorch for severity CNN (planned)
- HTML/CSS for web UI
- OpenStreetMap/Nominatim for reverse geocoding (planned)
- PostgreSQL/SQLite for persistence (planned)

## 📊 Model Details
- YOLOv8 trained on RDD2022 (single-class pothole).
- Current focus: pothole detection only; severity classification to be added.
- Achieved: mAP50 ≈ 0.81 (representative benchmark; update with latest evals).
- Domain shift: real-world lighting, camera angles, and regional textures can reduce precision/recall—mitigation via low `conf=0.05`, data augmentation, and future active learning.

## 🚧 Development Phases
### Phase 1: Detection System (COMPLETED)
- Dataset preparation (RDD2022 split/cleanup).
- YOLOv8 training and evaluation.
- Flask web app for image upload and detection with annotated outputs.

### Phase 2: Severity Classification (NEXT)
- Train CNN to classify pothole severity: Small / Medium / Severe.
- Integrate classifier post-detection with shared metadata payload.

### Phase 3: Backend API System
- Migrate to scalable FastAPI service.
- Endpoints: `/detect`, `/analyze` (severity + metadata), `/report` (complaint payload).

### Phase 4: Location Intelligence
- Capture GPS from client devices.
- Reverse geocode to addresses/districts (Kerala-first rollout).
- Persist geo-linked detections for mapping and analytics.

### Phase 5: Complaint Management System
- Generate structured complaints (templated email/PDF/API payloads).
- Optional authority integration when public APIs are available.

### Phase 6: Frontend Expansion
- Enhanced UI/UX with dashboards, analytics, heatmaps, and batch upload.
- Mobile-friendly flows; potential mobile app wrapper.

### Phase 7: Real-Time System
- Video ingestion with frame-wise detection.
- Edge deployment exploration (Jetson/NPU) for on-vehicle inference.

## 🔥 Key Features
- Real-time pothole detection with low confidence for high recall.
- Modular, layered architecture enabling incremental capability rollout.
- Designed for geo-aware reporting and complaint generation.
- Extensible to severity analysis, dashboards, and API-driven workflows.

## ⚠️ Challenges & Learnings
- **Domain shift**: Variation in lighting, weather, and road textures impacts precision/recall.
- **Dataset coverage**: Single-domain datasets may underrepresent regional conditions.
- **Generalization**: Requires augmentation, active learning, and periodic re-training.
- **Field vs. lab gap**: Real-world camera quality and motion blur require robust preprocessing.

## 🚀 Future Enhancements
- Mobile app integration and crowdsourced data collection.
- Smart prioritization using severity + traffic density + location risk.
- Government API integrations where available for automated ticketing.
- Active learning loop to ingest user feedback and improve models.

## 🧨 Project Vision Statement
"Transforming road damage detection into an intelligent, scalable civic infrastructure monitoring system using AI."
