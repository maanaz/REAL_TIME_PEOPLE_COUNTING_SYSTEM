import os
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

# Sidebar: 1. Video Input
st.sidebar.header("1. Video Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload a video file", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is None:
    st.info("Upload a video from the sidebar to begin.")
    st.stop()

# Reset confirmed line if a different video is uploaded
if st.session_state.get("current_video") != uploaded_file.name:
    st.session_state["current_video"] = uploaded_file.name
    st.session_state.pop("confirmed_line", None)

os.makedirs("videos", exist_ok=True)
video_path = os.path.join("videos", uploaded_file.name)

with open(video_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

vp = VideoProcessor(video_path)
meta = vp.get_metadata()
first_frame_rgb = vp.get_first_frame(as_rgb=True)
vp.release()

# Display Metadata
col1, col2, col3 = st.columns(3)
col1.metric("Resolution", f"{meta['width']} x {meta['height']}")
col2.metric("FPS", meta["fps"])
col3.metric("Total Frames", meta["total_frames"])

# Sidebar: 2. Line Calibration Mode
st.sidebar.header("2. Virtual Line Setup")
saved_preset = get_preset(uploaded_file.name)

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
        key=f"canvas_{uploaded_file.name}",
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
                save_preset(uploaded_file.name, candidate_line)
                st.session_state["confirmed_line"] = candidate_line
                st.toast(f"Preset saved for {uploaded_file.name}")

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

if st.button("Start People Counting", type="primary"):
    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(uploaded_file.name)[0]
    output_path = os.path.join("outputs", f"{base_name}_{tracker_choice.lower()}_counted.mp4")

    # Convert confirmed normalized line to full video pixel coordinates
    line_pt1, line_pt2 = denormalize_line(
        st.session_state["confirmed_line"], meta["width"], meta["height"]
    )

    vp_run = VideoProcessor(video_path)
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
        f"Running on Device: {'CUDA GPU (FP16)' if tracker.use_cuda else 'CPU'} | "
        f"Model: {model_choice} | Tracker: {tracker_choice}"
    )

    # Live UI Placeholders
    m_col1, m_col2, m_col3 = st.columns(3)
    metric_ab = m_col1.empty()
    metric_ba = m_col2.empty()
    metric_total = m_col3.empty()

    progress_bar = st.progress(0)
    frame_placeholder = st.empty()

    counts = counter.get_counts()
    total_frames = max(1, meta["total_frames"])

    try:
        for idx, frame_bgr in vp_run.stream_frames():
            tracks = tracker.update(frame_bgr)
            counts = counter.update(tracks, idx)
            annotated_bgr = visualizer.draw(frame_bgr, tracks, counts)

            writer.write(annotated_bgr)

            # Update live metrics & preview
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

    st.success(
        f"Counting Complete! A → B: {counts['A_TO_B']} | "
        f"B → A: {counts['B_TO_A']} | Total: {counts['TOTAL']}"
    )

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Annotated Video (.mp4)",
            data=f,
            file_name=os.path.basename(output_path),
            mime="video/mp4",
        )