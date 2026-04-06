# Smart Road Damage Detection & Complaint Intelligence System

## 🎯 Project Objective
- End-to-end AI platform to detect potholes and road damage using a **3-model YOLOv8 ensemble**, classify severity (planned CNN), capture user-uploaded road imagery, enrich with geolocation, and generate structured complaints for civic authorities.
- Provide a data-driven pipeline from image/video ingestion to actionable insights for maintenance teams.

## 🧠 Problem Statement
- Road infrastructure monitoring is reactive and manual, leading to delayed maintenance and higher repair costs.
- Existing complaint systems are cumbersome, unstructured, and lack evidence linking (image + location + severity).
- Real-time, geo-tagged reporting is scarce, creating a gap between citizens, field teams, and authorities.

## 💡 Solution Overview
- **Multi-model ensemble** detection via 3 YOLOv8s models trained with cascaded transfer learning for robust multi-POV coverage.
- Cross-model NMS merges predictions; detections agreed upon by 2+ models are flagged as high-confidence consensus.
- Planned severity classification (small/medium/severe) to prioritize repairs and resource allocation.
- Image-to-insight pipeline that links detections with GPS/reverse geocoding for location-aware reports.
- Future complaint automation to generate structured, authority-ready submissions (email/PDF/API when available).

## 🏗️ System Architecture
### Core Components
1. **AI Layer**: 3-model YOLOv8s ensemble with cross-model NMS and consensus tracking.
2. **Backend**: Flask (current) with a path to FastAPI for scalable APIs.
3. **Frontend**: Premium dark-mode web UI (upload, results, detection breakdown); mobile/web expansion planned.
4. **Location Intelligence**: GPS capture on client; reverse geocoding (OpenStreetMap/Nominatim); district mapping (Kerala focus initially).
5. **Complaint Engine**: Template-based, structured report generation; future integration with civic APIs.
6. **Database**: Stores uploads, detections, metadata, geo-coordinates, and complaint states.

### Pipeline Flow
```
Image → Model A (Dashcam) ─┐
      → Model B (Close-up) ─┤→ Cross-Model NMS → Consensus Detection →
      → Model C (Street)   ─┘   Severity Classification (planned) →
Location Enrichment → Structured Complaint → Dashboards & Reporting
```

## ⚙️ Technologies Used
- Python, Flask (current) / FastAPI (planned)
- YOLOv8s (Ultralytics) × 3 models, OpenCV, NumPy
- Cascaded transfer learning: COCO → Model A → Model B → Model C
- HTML/CSS (dark mode, glassmorphism) for web UI
- OpenStreetMap/Nominatim for reverse geocoding (planned)
- PostgreSQL/SQLite for persistence (planned)

## 📊 Model Details

### Multi-Model Ensemble Architecture

| Model | Weights | Dataset | POV | Strategy | Conf Threshold |
|-------|---------|---------|-----|----------|---------------|
| **Model A** | `model_a_rdd2022.pt` | RDD2022 (4 classes) | Dashcam / moving vehicle | COCO pretrained → train | 0.05 |
| **Model B** | `model_b_pothole600.pt` | Pothole-600 (1 class) | Close-up / inspection | Fine-tune from Model A | 0.10 |
| **Model C** | `model_c_kaggle.pt` | Kaggle Annotated (1 class) | Street / pedestrian | Fine-tune from Model B | 0.05 |

**Transfer chain:** `COCO → Model A → Model B → Model C`
Each model inherits road-damage knowledge from the previous one.

### Cross-Model NMS
- All 3 models run on every image at their individual confidence thresholds.
- Overlapping detections (IoU > 0.45) are merged, keeping the highest-confidence prediction.
- Agreeing models are tracked per detection — **consensus = high reliability signal**.
- Color-coded bounding boxes: 🔴 Red = Model A, 🟢 Green = Model B, 🔵 Blue = Model C.

### Damage Classes
- **Model A** detects: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), D40 (pothole)
- **Models B & C** detect: pothole (single class)

## 🚧 Development Phases
### Phase 1: Multi-Model Detection System (COMPLETED ✅)
- Dataset preparation (RDD2022, Pothole-600, Kaggle Annotated).
- Cascaded transfer learning: trained 3 YOLOv8s models.
- Cross-model NMS fusion pipeline.
- Premium dark-mode Flask web app with ensemble analysis, detection table, and consensus indicators.
- JSON API endpoint (`/api/detect`) for programmatic access.

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
- **3-model ensemble** with cross-model consensus for high-confidence detections.
- Color-coded bounding boxes per model with consensus-thickened borders.
- Detection breakdown table with per-detection confidence bars and model attribution.
- Premium dark-mode UI with glassmorphism and micro-animations.
- JSON API for programmatic integration.
- Modular, layered architecture enabling incremental capability rollout.
- Designed for geo-aware reporting and complaint generation.

## ⚠️ Challenges & Learnings
- **Domain shift**: Variation in lighting, weather, and road textures impacts precision/recall — mitigated by multi-POV training.
- **Dataset coverage**: Single-domain datasets underrepresent regional conditions — solved with 3-dataset ensemble.
- **Model consensus**: Multi-model agreement significantly reduces false positives.
- **Field vs. lab gap**: Real-world camera quality and motion blur require robust preprocessing.

## 🚀 Future Enhancements
- Mobile app integration and crowdsourced data collection.
- Smart prioritization using severity + traffic density + location risk.
- Government API integrations where available for automated ticketing.
- Active learning loop to ingest user feedback and improve models.

## 🧨 Project Vision Statement
"Transforming road damage detection into an intelligent, scalable civic infrastructure monitoring system using AI."
