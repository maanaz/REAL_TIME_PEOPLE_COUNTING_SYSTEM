import json
import os
AVAILABLE_MODELS = ["yolo26s.pt", "yolo26n.pt", "yolo26m.pt"]
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_CONF = 0.40

TRACKER_CONFIGS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}
PRESETS_PATH = os.path.join("assets", "presets.json")


def load_all_presets() -> dict:
    """Loads all saved line presets from assets/presets.json."""
    if not os.path.exists(PRESETS_PATH):
        return {}
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_preset(preset_name: str, norm_line: list) -> None:
    """Saves a normalized line [[nx1, ny1], [nx2, ny2]] under preset_name."""
    os.makedirs(os.path.dirname(PRESETS_PATH), exist_ok=True)
    presets = load_all_presets()
    presets[preset_name] = norm_line
    with open(PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def get_preset(preset_name: str):
    """Returns a saved normalized line if it exists, else None."""
    presets = load_all_presets()
    return presets.get(preset_name)