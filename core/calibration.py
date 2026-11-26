import cv2
import numpy as np
from PIL import Image


def resize_for_canvas(frame_rgb: np.ndarray, max_width: int = 700):
    """Resizes an RGB frame for the Streamlit canvas while preserving aspect ratio."""
    orig_h, orig_w = frame_rgb.shape[:2]
    scale = max_width / orig_w
    canvas_w = max_width
    canvas_h = int(orig_h * scale)

    resized = cv2.resize(frame_rgb, (canvas_w, canvas_h))
    pil_image = Image.fromarray(resized)
    return pil_image, canvas_w, canvas_h


def extract_normalized_line(canvas_json: dict, canvas_w: int, canvas_h: int):
    """
    Extracts the last drawn line from streamlit-drawable-canvas JSON
    and returns normalized coordinates: [(nx1, ny1), (nx2, ny2)].
    """
    if not canvas_json or "objects" not in canvas_json:
        return None

    # Filter only line objects and grab the most recent one
    lines = [obj for obj in canvas_json["objects"] if str(obj.get("type", "")).lower() == "line"]
    if not lines:
        return None

    line = lines[-1]

    # Fabric.js stores line endpoints relative to the object's center (left, top)
    cx = line["left"]
    cy = line["top"]
    x1 = cx + line["x1"]
    y1 = cy + line["y1"]
    x2 = cx + line["x2"]
    y2 = cy + line["y2"]

    # Clamp to canvas boundaries and normalize to 0.0 - 1.0
    nx1 = max(0.0, min(1.0, x1 / canvas_w))
    ny1 = max(0.0, min(1.0, y1 / canvas_h))
    nx2 = max(0.0, min(1.0, x2 / canvas_w))
    ny2 = max(0.0, min(1.0, y2 / canvas_h))

    return [(round(nx1, 4), round(ny1, 4)), (round(nx2, 4), round(ny2, 4))]


def get_auto_line(orientation: str = "Horizontal (Center)"):
    """Returns normalized coordinates for automatic default lines."""
    if orientation == "Vertical (Center)":
        return [(0.5, 0.1), (0.5, 0.9)]
    # Default: Horizontal across the middle
    return [(0.1, 0.5), (0.9, 0.5)]


def denormalize_line(norm_line, frame_w: int, frame_h: int):
    """Converts normalized [(nx1, ny1), (nx2, ny2)] to integer pixel coordinates."""
    pt1 = (int(norm_line[0][0] * frame_w), int(norm_line[0][1] * frame_h))
    pt2 = (int(norm_line[1][0] * frame_w), int(norm_line[1][1] * frame_h))
    return pt1, pt2


def draw_preview_line(frame_rgb: np.ndarray, norm_line):
    """Draws the confirmed/auto line on a copy of the RGB frame for preview."""
    preview = frame_rgb.copy()
    h, w = preview.shape[:2]
    pt1, pt2 = denormalize_line(norm_line, w, h)

    cv2.line(preview, pt1, pt2, (255, 0, 0), 3)  # Red line in RGB
    cv2.circle(preview, pt1, 6, (0, 255, 0), -1)
    cv2.circle(preview, pt2, 6, (0, 255, 0), -1)
    cv2.putText(preview, "A", (pt1[0] + 8, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(preview, "B", (pt2[0] + 8, pt2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return preview