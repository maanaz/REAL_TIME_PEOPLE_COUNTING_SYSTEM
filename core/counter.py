class LineCounter:
    """
    Tracks directional crossings (A -> B and B -> A) while keeping TOTAL
    strictly equal to the number of unique track IDs that crossed at least once.
    """

    def __init__(
        self,
        line_pt1: tuple[int, int],
        line_pt2: tuple[int, int],
        cooldown_frames: int = 30,
        min_track_frames: int = 3,
        min_travel_px: float = 15.0,
    ):
        self.pt_a = line_pt1
        self.pt_b = line_pt2
        self.cooldown_frames = cooldown_frames
        self.min_track_frames = min_track_frames
        self.min_travel_px = min_travel_px

        self.track_history: dict[int, list[tuple[int, int]]] = {}
        self.crossing_state: dict[int, tuple[str, int]] = {}
        self.unique_crossed_ids: set[int] = set()
        self.recently_crossed_ids: set[int] = set()

        self.count_a_to_b = 0
        self.count_b_to_a = 0

    @staticmethod
    def _ccw(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    def _segments_intersect(self, p_prev: tuple[int, int], p_curr: tuple[int, int]) -> bool:
        return (
            self._ccw(p_prev, self.pt_a, self.pt_b) != self._ccw(p_curr, self.pt_a, self.pt_b)
            and self._ccw(p_prev, p_curr, self.pt_a) != self._ccw(p_prev, p_curr, self.pt_b)
        )

    def _get_crossing_direction(self, p_prev: tuple[int, int], p_curr: tuple[int, int]) -> str | None:
        ax, ay = self.pt_a
        bx, by = self.pt_b

        cross_prev = (bx - ax) * (p_prev[1] - ay) - (by - ay) * (p_prev[0] - ax)
        cross_curr = (bx - ax) * (p_curr[1] - ay) - (by - ay) * (p_curr[0] - ax)

        if cross_prev < 0 and cross_curr >= 0:
            return "A_TO_B"
        elif cross_prev > 0 and cross_curr <= 0:
            return "B_TO_A"
        return None

    def update(self, tracks: list[dict], frame_idx: int) -> dict:
        self.recently_crossed_ids.clear()

        for track in tracks:
            track_id = track["track_id"]
            curr_pos = track["anchor"]

            history = self.track_history.setdefault(track_id, [])
            history.append(curr_pos)

            if len(history) < self.min_track_frames:
                continue

            # Ignore stationary jitter on the line
            start_pos = history[0]
            net_travel = ((curr_pos[0] - start_pos[0]) ** 2 + (curr_pos[1] - start_pos[1]) ** 2) ** 0.5
            if net_travel < self.min_travel_px:
                continue

            prev_pos = history[-2]

            if self._segments_intersect(prev_pos, curr_pos):
                direction = self._get_crossing_direction(prev_pos, curr_pos)
                last_dir, last_frame = self.crossing_state.get(track_id, (None, -9999))

                # Require cooldown even on reverse direction so jitter never triggers a return count
                cooldown_elapsed = (frame_idx - last_frame) > self.cooldown_frames

                if direction is not None and cooldown_elapsed and direction != last_dir:
                    if direction == "A_TO_B":
                        self.count_a_to_b += 1
                    elif direction == "B_TO_A":
                        self.count_b_to_a += 1

                    self.crossing_state[track_id] = (direction, frame_idx)
                    self.unique_crossed_ids.add(track_id)  # Set automatically deduplicates IDs
                    self.recently_crossed_ids.add(track_id)

        return self.get_counts()

    def get_counts(self) -> dict:
        return {
            "A_TO_B": self.count_a_to_b,
            "B_TO_A": self.count_b_to_a,
            "TOTAL": len(self.unique_crossed_ids),
            "crossed_now": self.recently_crossed_ids,
        }