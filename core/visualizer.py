import cv2
import numpy as np


class FrameVisualizer:
    """Draws the virtual line, Side A / Side B zone labels, bounding boxes, and HUD."""

    def __init__(self, line_pt1: tuple[int, int], line_pt2: tuple[int, int]):
        self.pt_a = line_pt1
        self.pt_b = line_pt2

        # Precompute midpoint and unit normal vector
        ax, ay = line_pt1
        bx, by = line_pt2
        self.mid = (int((ax + bx) / 2), int((ay + by) / 2))

        dx, dy = bx - ax, by - ay
        length = max(1.0, np.hypot(dx, dy))
        # +normal points into Side B (positive cross product); -normal points into Side A
        self.nx = -dy / length
        self.ny = dx / length

    @staticmethod
    def _draw_badge(canvas: np.ndarray, text: str, center: tuple[int, int], text_color: tuple[int, int, int]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        cx, cy = center
        x1 = max(5, cx - tw // 2 - 8)
        y1 = max(5, cy - th // 2 - 6)
        x2 = min(canvas.shape[1] - 5, cx + tw // 2 + 8)
        y2 = min(canvas.shape[0] - 5, cy + th // 2 + 6)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        cv2.putText(
            canvas, text, (x1 + 8, y2 - 6),
            font, scale, text_color, thickness, cv2.LINE_AA
        )

    def _draw_side_labels(self, canvas: np.ndarray):
        mx, my = self.mid
        offset = 45

        # Side A (negative cross-product side)
        pos_a = (int(mx - self.nx * offset), int(my - self.ny * offset))
        self._draw_badge(canvas, "Side A", pos_a, (0, 255, 0))

        # Side B (positive cross-product side)
        pos_b = (int(mx + self.nx * offset), int(my + self.ny * offset))
        self._draw_badge(canvas, "Side B", pos_b, (0, 200, 255))

    def draw(self, frame_bgr: np.ndarray, tracks: list[dict], counts: dict) -> np.ndarray:
        canvas = frame_bgr.copy()
        crossed_now = counts.get("crossed_now", set())

        # 1. Draw Virtual Line & Side A / Side B Labels
        cv2.line(canvas, self.pt_a, self.pt_b, (0, 0, 255), 3)
        self._draw_side_labels(canvas)

        # 2. Draw Tracked People (Flash bright yellow on the exact frame they cross)
        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            anchor = t["anchor"]

            just_crossed = tid in crossed_now
            box_color = (0, 255, 255) if just_crossed else (0, 220, 0)
            thickness = 3 if just_crossed else 2

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
            cv2.circle(canvas, anchor, 4, (255, 0, 255), -1)
            cv2.putText(
                canvas,
                f"ID:{tid}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )

        # 3. Top-Left Semi-Transparent HUD Overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (15, 15), (300, 130), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        cv2.putText(canvas, f"A -> B : {counts['A_TO_B']}", (28, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"B -> A : {counts['B_TO_A']}", (28, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"TOTAL  : {counts['TOTAL']}", (28, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas