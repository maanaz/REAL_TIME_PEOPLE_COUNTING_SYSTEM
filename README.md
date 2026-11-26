# 🚶 People Counter — Bidirectional Line-Crossing Analytics

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

A real-time computer vision system for bidirectional footfall analytics (**Side A → Side B**, **Side B → Side A**, and **Unique Total**) across a configurable virtual tripwire—powered by **YOLO26**, **BoT-SORT / ByteTrack**, **OpenCV**, and **Streamlit**.

---

## ✨ Key Features

- **Interactive Tripwire Configuration:** Define and save custom counting lines directly on the first video frame, complete with clear **Side A** and **Side B** zone badges.
- **Persistent Multi-Object Tracking:** Powered by Ultralytics **YOLO26** (`yolo26n.pt` / `yolo26s.pt`) with **BoT-SORT** or **ByteTrack** to maintain stable person IDs across frames.
- **Jitter & Duplicate Protection (`v2`):**
  - Exact 2D vector cross-product segment intersection evaluated at each person's bottom-center foot anchor.
  - Trajectory stabilization (`min_track_frames`) and net displacement filtering (`min_travel_px`) prevent stationary line jitter from inflating counts.
  - Per-track-ID directional cooldown and unique ID tracking ensure `TOTAL` reflects unique individuals while directional counters capture genuine round trips.
- **Annotated Video Export:** Exports full-resolution `.mp4` outputs with bounding boxes, track IDs, crossing flashes, and live HUD metrics.
- **Zero-Setup Docker Deployment:** Pre-built image on Docker Hub plus a portable `Dockerfile` for instant execution across Windows, macOS, and Linux.

---

## 🚀 Ways to Run

### Option 1: Instant Run via Docker Hub (Fastest — No Clone Required)

Pull and launch the pre-built container directly from Docker Hub (`maanazkantony/people-counter:v1`):

    docker run --rm -p 8501:8501 --name people-counter-app maanazkantony/people-counter:v1

Open **http://localhost:8501** in your browser.

---

### Option 2: Build Locally from Dockerfile

1. **Clone the repository:**

       git clone -b v1 https://github.com/maanazkantony/PEOPLE_COUNTER.git
       cd PEOPLE_COUNTER

2. **Build and run the Docker image:**

       docker build -t people-counter:v1 .
       docker run --rm -p 8501:8501 --name people-counter-app people-counter:v1

3. Open **http://localhost:8501** in your browser *(use `localhost` rather than the external IP printed in the console)*.

---

### Option 3: Run Locally via GitHub & Python

1. **Clone the repository and select a branch (`v1` or `v2`):**

       git clone -b v2 https://github.com/maanazkantony/PEOPLE_COUNTER.git
       cd PEOPLE_COUNTER

2. **Create and activate a virtual environment:**

       python -m venv venv
       # Windows PowerShell:
       .\venv\Scripts\Activate.ps1
       # Linux / macOS:
       source venv/bin/activate

3. **Install dependencies and launch Streamlit:**

       pip install -r requirements.txt
       streamlit run app.py

4. Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

    PEOPLE_COUNTER/
    ├── app.py                 # Streamlit web UI and video processing pipeline
    ├── core/
    │   ├── counter.py         # Tripwire intersection geometry, direction logic, and ID deduplication
    │   ├── tracker.py         # YOLO26 + BoT-SORT/ByteTrack inference wrapper
    │   └── visualizer.py      # Bounding boxes, Side A/B badges, tripwire, and HUD overlays
    ├── utils/                 # Video I/O and helper utilities
    ├── assets/
    │   └── presets.json       # Saved tripwire coordinate presets
    ├── models/                # YOLO26 weights (.pt)
    ├── videos/                # Input video directory
    ├── outputs/               # Exported annotated .mp4 videos
    ├── Dockerfile             # Portable Linux container definition
    ├── .dockerignore
    └── requirements.txt

---

## ⚙️ Counter Parameters (`core/counter.py`)

| Parameter | Default | Purpose |
| :--- | :---: | :--- |
| `min_track_frames` | `3` | Minimum consecutive frames a track ID must exist before registering a crossing |
| `min_travel_px` | `15.0` | Minimum net displacement from initial detection to ignore stationary line wobble |
| `cooldown_frames` | `30` | Minimum frame gap required before the same ID can register a reverse crossing |

---

## 🏷️ Release Branches

- **`v1`**: Production baseline with interactive tripwire setup, Side A/B zone badges, annotated `.mp4` export, and Docker Hub image (`maanazkantony/people-counter:v1`).
- **`v2`**: Upgraded counting engine with unique `track_id` deduplication, net-displacement jitter filtering, and clean `FP32` inference.