import os
import torch
from ultralytics import YOLO
from utils.config import DEFAULT_MODEL, DEFAULT_CONF, TRACKER_CONFIGS


class PersonTracker:
    """Handles YOLO person detection + switchable ByteTrack / BoT-SORT tracking."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tracker_name: str = "ByteTrack",
        conf_threshold: float = DEFAULT_CONF,
    ):
        self.conf_threshold = conf_threshold
        self.tracker_yaml = TRACKER_CONFIGS.get(tracker_name, "bytetrack.yaml")

        # Keep downloaded weights organized inside models/
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", model_name)
        self.model = YOLO(model_path if os.path.exists(model_path) else model_name)

        # Hardware & VRAM check: use GPU only if CUDA is available and >= 1.5 GB VRAM is free
        self.use_cuda = False
        if torch.cuda.is_available():
            free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
            if free_vram_gb >= 1.5:
                self.use_cuda = True

        self.device = 0 if self.use_cuda else "cpu"

    def update(self, frame_bgr) -> list[dict]:
        """
        Runs detection + tracking on a single BGR frame.
        Returns a list of tracked people:
        [{'track_id': int, 'bbox': (x1, y1, x2, y2), 'anchor': (ax, ay), 'conf': float}, ...]
        """
        results = self.model.track(
            source=frame_bgr,
            persist=True,
            classes=[0],  # Class 0 = 'person' in COCO
            tracker=self.tracker_yaml,
            conf=self.conf_threshold,
            device=self.device,
            half=self.use_cuda,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            return detections

        xyxy_list = boxes.xyxy.cpu().numpy()
        ids_list = boxes.id.int().cpu().tolist()
        confs_list = boxes.conf.cpu().tolist()

        for bbox, track_id, conf in zip(xyxy_list, ids_list, confs_list):
            x1, y1, x2, y2 = map(int, bbox)
            # Bottom-center anchor (foot position on the floor plane)
            anchor = (int((x1 + x2) / 2), y2)

            detections.append(
                {
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "anchor": anchor,
                    "conf": round(conf, 2),
                }
            )

        return detections