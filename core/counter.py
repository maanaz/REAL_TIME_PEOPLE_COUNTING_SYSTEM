class LineCounter:
    """
    Tracks foot-point trajectories across a finite line segment (A -> B)
    and counts crossings in both directions without duplicate triggers.
    """

    def __init__(self, line_pt1: tuple[int, int], line_pt2: tuple[int, int], cooldown_frames: int = 30):
        self.pt_a = line_pt1
        self.pt_b = line_pt2
        self.cooldown_frames = cooldown_frames

        self.track_history: dict[int, tuple[int, int]] = {}
        self.crossing_state: dict[int, tuple[str, int]] = {}
        self.recently_crossed_ids: set[int] = set()

        self.count_a_to_b = 0
        self.count_b_to_a = 0

    @staticmethod
    def _ccw(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    def _segments_intersect(self, p_prev: tuple[int, int], p_curr: tuple[int, int]) -> bool:
        """Checks if movement vector p_prev -> p_curr intersects finite segment pt_a -> pt_b."""
        return (
            self._ccw(p_prev, self.pt_a, self.pt_b) != self._ccw(p_curr, self.pt_a, self.pt_b)
            and self._ccw(p_prev, p_curr, self.pt_a) != self._ccw(p_prev, p_curr, self.pt_b)
        )

    def _get_crossing_direction(self, p_prev: tuple[int, int], p_curr: tuple[int, int]) -> str | None:
        """Uses 2D cross product relative to vector A -> B to determine crossing side."""
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
        """
        Updates counts given tracked detections for the current frame.
        Returns current counts and IDs that triggered a crossing on this frame.
        """
        self.recently_crossed_ids.clear()

        for track in tracks:
            track_id = track["track_id"]
            curr_pos = track["anchor"]

            if track_id in self.track_history:
                prev_pos = self.track_history[track_id]

                if self._segments_intersect(prev_pos, curr_pos):
                    direction = self._get_crossing_direction(prev_pos, curr_pos)
                    last_dir, last_frame = self.crossing_state.get(track_id, (None, -9999))

                    is_new_direction = direction != last_dir
                    cooldown_elapsed = (frame_idx - last_frame) > self.cooldown_frames

                    if direction is not None and (is_new_direction or cooldown_elapsed):
                        if direction == "A_TO_B":
                            self.count_a_to_b += 1
                        elif direction == "B_TO_A":
                            self.count_b_to_a += 1

                        self.crossing_state[track_id] = (direction, frame_idx)
                        self.recently_crossed_ids.add(track_id)

            self.track_history[track_id] = curr_pos

        return self.get_counts()

    def get_counts(self) -> dict:
        return {
            "A_TO_B": self.count_a_to_b,
            "B_TO_A": self.count_b_to_a,
            "TOTAL": self.count_a_to_b + self.count_b_to_a,
            "crossed_now": self.recently_crossed_ids,
        }