# People Counter — Bidirectional Line-Crossing Analytics

A real-time computer vision application for counting directional footfall (`Side A → Side B`, `Side B → Side A`, and `Unique Total`) across a configurable virtual tripwire using **YOLO26**, **BoT-SORT / ByteTrack**, **OpenCV**, and **Streamlit**.

---

## Features

- **Interactive Virtual Tripwire:** Configure and save custom counting lines directly on the first video frame, with clear **Side A** and **Side B** directional zone badges.
- **Multi-Object Person Tracking:** Uses Ultralytics YOLO26 (`yolo26n.pt` / `yolo26s.pt`) with BoT-SORT or ByteTrack to maintain persistent track IDs across video frames.
- **Line-Crossing Geometry & Jitter Protection (`v2`):**
  - Exact 2D vector cross-product segment intersection evaluated at each person's bottom-center foot anchor.
  - Minimum trajectory length (`min_track_frames`) and net displacement filtering (`min_travel_px`) eliminate false counts from stationary bounding-box jitter on the line.
  - Per-track-ID directional cooldown and unique ID tracking ensure `TOTAL` reflects unique individuals while `Side A → Side B` and `Side B → Side A` capture genuine round trips.
- **Annotated Video Export:** Renders and exports full-resolution `.mp4` videos with bounding boxes, track IDs, crossing highlight flashes, and live count HUD overlays.
- **Containerized Deployment:** Includes a portable CPU-compatible `Dockerfile` so the app runs cleanly on Windows, macOS, and Linux without manual CUDA or OpenCV setup.

---

## Project Structure

    PEOPLE_COUNTER/
    ├── app.py                 # Streamlit web interface and frame processing loop
    ├── core/
    │   ├── counter.py         # Tripwire intersection geometry, direction logic, and ID deduplication
    │   ├── tracker.py         # YOLO26 + BoT-SORT/ByteTrack inference wrapper
    │   └── visualizer.py      # Bounding boxes, Side A/B badges, tripwire, and HUD overlays
    ├── utils/                 # Video reader/writer and helper utilities
    ├── assets/
    │   └── presets.json       # Saved tripwire coordinate presets
    ├── models/                # YOLO26 model weights (.pt)
    ├── videos/                # Input video directory
    ├── outputs/               # Exported annotated .mp4 videos
    ├── Dockerfile             # Portable Linux container definition
    ├── .dockerignore
    └── requirements.txt

---

## Quick Start (Local Python)

1. **Clone the repository and select a branch (`v1` or `v2`):**

       git clone -b v2 https://github.com/<your-username>/PEOPLE_COUNTER.git
       cd PEOPLE_COUNTER

2. **Create and activate a virtual environment:**

       python -m venv venv
       # Windows PowerShell:
       .\venv\Scripts\Activate.ps1
       # Linux / macOS:
       source venv/bin/activate

3. **Install dependencies and run the app:**

       pip install -r requirements.txt
       streamlit run app.py

4. Open **http://localhost:8501** in your browser.

---

## Run with Docker

Build and run the containerized application on any machine with Docker Desktop installed:

    docker build -t people-counter:v1 .
    docker run --rm -p 8501:8501 --name people-counter-app people-counter:v1

Open **http://localhost:8501** in your browser *(use `localhost` rather than the external IP address printed in the terminal)*.

---

## Configuration & Tuning (`core/counter.py`)

- `min_track_frames` *(default: `3`)*: Minimum consecutive frames a track ID must exist before triggering a crossing.
- `min_travel_px` *(default: `15.0`)*: Minimum net pixel displacement from a track's starting point to filter out stationary line wobble.
- `cooldown_frames` *(default: `30`)*: Minimum frame gap required before the same track ID can register a reverse-direction crossing.

---

## Release Branches

- **`v1`**: Baseline release with interactive tripwire configuration, Side A/B zone badges, annotated `.mp4` export, and Docker support.
- **`v2`**: Upgraded counting engine with unique `track_id` deduplication, net-displacement jitter filtering, and clean `FP32` inference.