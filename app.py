import os
import time
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from core.video_processor import VideoProcessor
from core.calibration import (
    resize_for_canvas,
    extract_normalized_line,
    get_auto_line,
    denormalize_line,
    draw_preview_line,
)
from core.tracker import PersonTracker
from core.counter import LineCounter
from core.visualizer import FrameVisualizer
from utils.config import (
    AVAILABLE_MODELS,
    TRACKER_CONFIGS,
    DEFAULT_CONF,
    save_preset,
    get_preset,
)

st.set_page_config(page_title="People Counter", layout="wide")
st.title("Real-Time People Counter")

# ==========================================================
# Sidebar: 1. Video / Camera Input
# ==========================================================
st.sidebar.header("1. Input Source")
source_type = st.sidebar.radio(
    "Select Source",
    ["Upload Video", "Webcam (Local)", "IP Camera / RTSP"],
)

video_source = None
source_id = None
is_live = False

if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is None:
        st.info("Upload a video from the sidebar to begin.")
        st.stop()

    os.makedirs("videos", exist_ok=True)
    video_source = os.path.join("videos", uploaded_file.name)
    with open(video_source, "wb") as f:
        f.write(uploaded_file.getbuffer())
    source_id = uploaded_file.name

elif source_type == "Webcam (Local)":
    cam_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    video_source = int(cam_index)
    source_id = f"webcam_{cam_index}"
    is_live = True

elif source_type == "IP Camera / RTSP":
    stream_url = st.sidebar.text_input(
        "Stream URL",
        placeholder="rtsp://user:pass@192.168.1.50:554/stream or http://192.168.1.5:8080/video",
    )
    if not stream_url.strip():
        st.info("Enter an RTSP or HTTP IP camera URL in the sidebar to begin.")
        st.stop()
    video_source = stream_url.strip()
    source_id = f"ipcam_{abs(hash(video_source)) % 100000}"
    is_live = True

# Reset confirmed line & cached preview frame if source changes
if st.session_state.get("current_source_id") != source_id:
    st.session_state["current_source_id"] = source_id
    st.session_state.pop("confirmed_line", None)
    st.session_state.pop("live_preview_frame", None)
    st.session_state.pop("live_meta", None)

# ==========================================================
# Load Metadata & First Frame for Calibration
# ==========================================================
if not is_live:
    vp = VideoProcessor(video_source)
    meta = vp.get_metadata()
    first_frame_rgb = vp.get_first_frame(as_rgb=True)
    vp.release()
else:
    refresh_clicked = st.sidebar.button("Refresh Camera Frame")
    if refresh_clicked or "live_preview_frame" not in st.session_state:
        cap_preview = cv2.VideoCapture(video_source)
        if not cap_preview.isOpened():
            st.error("Could not open camera/stream. Check the camera index or RTSP URL.")
            st.stop()

        ret, frame_bgr = cap_preview.read()
        width = int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_preview.get(cv2.CAP_PROP_FPS)
        cap_preview.release()

        if not ret or frame_bgr is None:
            st.error("Connected to source, but failed to grab a frame.")
            st.stop()

        st.session_state["live_preview_frame"] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["live_meta"] = {
            "width": width if width > 0 else frame_bgr.shape[1],
            "height": height if height > 0 else frame_bgr.shape[0],
            "fps": round(fps, 2) if fps and fps > 0 else 20.0,
            "total_frames": 0,
        }

    first_frame_rgb = st.session_state["live_preview_frame"]
    meta = st.session_state["live_meta"]

# Display Metadata
col1, col2, col3 = st.columns(3)
col1.metric("Resolution", f"{meta['width']} x {meta['height']}")
col2.metric("FPS", meta["fps"])
col3.metric("Total Frames" if not is_live else "Stream Mode", meta["total_frames"] if not is_live else "Live")

# ==========================================================
# Sidebar: 2. Line Calibration Mode
# ==========================================================
st.sidebar.header("2. Virtual Line Setup")
saved_preset = get_preset(source_id)

mode_options = ["Manual (Draw on Frame)", "Auto (Default Line)"]
if saved_preset is not None:
    mode_options.append("Saved Preset")

line_mode = st.sidebar.radio("Calibration Mode", mode_options)

candidate_line = None

if line_mode == "Manual (Draw on Frame)":
    st.subheader("Step 1: Draw Virtual Line (Click and drag from A to B)")
    bg_image, canvas_w, canvas_h = resize_for_canvas(first_frame_rgb, max_width=700)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=bg_image,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="line",
        key=f"canvas_{source_id}",
    )

    if canvas_result.json_data is not None:
        candidate_line = extract_normalized_line(
            canvas_result.json_data, canvas_w, canvas_h
        )

elif line_mode == "Auto (Default Line)":
    orientation = st.sidebar.selectbox(
        "Auto Line Orientation",
        ["Horizontal (Center)", "Vertical (Center)"],
    )
    candidate_line = get_auto_line(orientation)
    st.subheader("Step 1: Auto Line Preview")
    st.image(draw_preview_line(first_frame_rgb, candidate_line), width=700)

elif line_mode == "Saved Preset":
    candidate_line = saved_preset
    st.subheader("Step 1: Saved Preset Preview")
    st.image(draw_preview_line(first_frame_rgb, candidate_line), width=700)

# Confirm & Save Controls
if candidate_line is not None:
    pt1, pt2 = denormalize_line(candidate_line, meta["width"], meta["height"])
    st.caption(f"Normalized: {candidate_line} | Video Pixels: A={pt1}, B={pt2}")

    if line_mode == "Saved Preset":
        if st.button("Confirm Virtual Line", type="primary"):
            st.session_state["confirmed_line"] = candidate_line
    else:
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("Confirm Virtual Line", type="primary"):
                st.session_state["confirmed_line"] = candidate_line
        with btn_col2:
            if st.button("Confirm & Save Preset"):
                save_preset(source_id, candidate_line)
                st.session_state["confirmed_line"] = candidate_line
                st.toast(f"Preset saved for {source_id}")

# ==========================================================
# STRICT PIPELINE GATE: Nothing below runs until line is set
# ==========================================================
if "confirmed_line" not in st.session_state:
    st.warning("Draw or select a virtual line and click 'Confirm Virtual Line' to unlock counting.")
    st.stop()

st.divider()
st.subheader("Step 2: Model & Tracker Configuration")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
with cfg_col1:
    model_choice = st.selectbox("YOLO Model", AVAILABLE_MODELS)
with cfg_col2:
    tracker_choice = st.selectbox("Tracker Algorithm", list(TRACKER_CONFIGS.keys()))
with cfg_col3:
    conf_thresh = st.slider("Confidence Threshold", 0.15, 0.85, DEFAULT_CONF, 0.05)

start_label = "Start Live Counting" if is_live else "Start People Counting"

if st.button(start_label, type="primary"):
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(source_id)[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S") if is_live else ""
    suffix = f"_{timestamp}" if timestamp else ""
    output_path = os.path.join("outputs", f"{base_name}_{tracker_choice.lower()}{suffix}_counted.mp4")

    line_pt1, line_pt2 = denormalize_line(
        st.session_state["confirmed_line"], meta["width"], meta["height"]
    )

    tracker = PersonTracker(
        model_name=model_choice,
        tracker_name=tracker_choice,
        conf_threshold=conf_thresh,
    )
    counter = LineCounter(
        line_pt1, line_pt2, cooldown_frames=max(15, int(meta["fps"]))
    )
    visualizer = FrameVisualizer(line_pt1, line_pt2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, meta["fps"], (meta["width"], meta["height"]))

    st.caption(
        f"Running on Device: {'CUDA GPU (FP32)' if tracker.use_cuda else 'CPU'} | "
        f"Model: {model_choice} | Tracker: {tracker_choice}"
    )

    if is_live:
        st.button("Stop Live Stream")

    # Live UI Placeholders
    m_col1, m_col2, m_col3 = st.columns(3)
    metric_ab = m_col1.empty()
    metric_ba = m_col2.empty()
    metric_total = m_col3.empty()

    progress_bar = st.progress(0) if not is_live else None
    frame_placeholder = st.empty()

    counts = counter.get_counts()

    if not is_live:
        vp_run = VideoProcessor(video_source)
        total_frames = max(1, meta["total_frames"])
        try:
            for idx, frame_bgr in vp_run.stream_frames():
                tracks = tracker.update(frame_bgr)
                counts = counter.update(tracks, idx)
                annotated_bgr = visualizer.draw(frame_bgr, tracks, counts)

                writer.write(annotated_bgr)

                metric_ab.metric("A → B", counts["A_TO_B"])
                metric_ba.metric("B → A", counts["B_TO_A"])
                metric_total.metric("Total Crossings", counts["TOTAL"])

                progress_bar.progress(min(1.0, (idx + 1) / total_frames))
                frame_placeholder.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), width=750
                )
        finally:
            vp_run.release()
            writer.release()
    else:
        cap_live = cv2.VideoCapture(video_source)
        if not cap_live.isOpened():
            writer.release()
            st.error("Failed to open live camera/stream.")
            st.stop()

        idx = 0
        try:
            while cap_live.isOpened():
                ret, frame_bgr = cap_live.read()
                if not ret or frame_bgr is None:
                    break

                tracks = tracker.update(frame_bgr)
                counts = counter.update(tracks, idx)
                annotated_bgr = visualizer.draw(frame_bgr, tracks, counts)

                writer.write(annotated_bgr)

                metric_ab.metric("A → B", counts["A_TO_B"])
                metric_ba.metric("B → A", counts["B_TO_A"])
                metric_total.metric("Total Crossings", counts["TOTAL"])

                frame_placeholder.image(
                    cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), width=750
                )
                idx += 1
        finally:
            cap_live.release()
            writer.release()

    st.success(
        f"Counting Complete! A → B: {counts['A_TO_B']} | "
        f"B → A: {counts['B_TO_A']} | Total: {counts['TOTAL']}"
    )

    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Annotated Video (.mp4)",
                data=f,
                file_name=os.path.basename(output_path),
                mime="video/mp4",
            )