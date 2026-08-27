import numpy as np
import math
import cv2
from datetime import datetime

class TrafficViolationEngine:
    def __init__(self, config):
        self.camera_id = config.get("camera_id", "CAM_01")
        self.stop_line = config["stop_line"]
        self.lane_cfg = config["lane_direction"]
        
        # Track histories: track_id -> list of (x_center, y_center, timestamp)
        self.track_history = {}
        # Logged violations to prevent multi-alerts for the same vehicle
        self.flagged_violations = set()

    def _get_direction_angle(self, p1, p2):
        """Calculates trajectory angle in degrees from p1 to p2 (0-360)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        rads = math.atan2(dy, dx)
        degs = math.degrees(rads)
        return (degs + 360) % 360

    def _line_intersect(self, p1, p2, q1, q2):
        """Checks if vehicle movement line segment (p1->p2) crosses stop line (q1->q2)."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    def process_tracks(self, tracks, is_red_light=False):
        """
        Evaluates active SORT tracks for Red-Light or Wrong-Way violations.
        tracks format: [[x1, y1, x2, y2, track_id], ...]
        """
        violations = []

        for trk in tracks:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)
            cx, cy = int((x1 + x2) / 2), int(y2)  # Bottom center of bounding box

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append((cx, cy))
            
            # Keep history to last 15 points
            if len(self.track_history[track_id]) > 15:
                self.track_history[track_id].pop(0)

            pts = self.track_history[track_id]
            if len(pts) < 3:
                continue

            prev_pt = pts[-2]
            curr_pt = pts[-1]
            start_pt = pts[0]

            # 1. Red-Light Violation Check
            if is_red_light and (track_id, "RED_LIGHT") not in self.flagged_violations:
                q1 = tuple(self.stop_line["pt1"])
                q2 = tuple(self.stop_line["pt2"])
                if self._line_intersect(prev_pt, curr_pt, q1, q2):
                    self.flagged_violations.add((track_id, "RED_LIGHT"))
                    violations.append({
                        "type": "RED_LIGHT_VIOLATION",
                        "track_id": track_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "camera_id": self.camera_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

            # 2. Wrong-Way / Opposite Direction Check
            displacement = math.hypot(curr_pt[0] - start_pt[0], curr_pt[1] - start_pt[1])
            if displacement >= self.lane_cfg["min_displacement_px"] and (track_id, "WRONG_WAY") not in self.flagged_violations:
                traj_angle = self._get_direction_angle(start_pt, curr_pt)
                allowed_angle = self.lane_cfg["allowed_angle_deg"]
                angle_diff = abs((traj_angle - allowed_angle + 180) % 360 - 180)

                if angle_diff > self.lane_cfg["angle_tolerance_deg"]:
                    self.flagged_violations.add((track_id, "WRONG_WAY"))
                    violations.append({
                        "type": "WRONG_WAY_VIOLATION",
                        "track_id": track_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "camera_id": self.camera_id,
                        "heading_angle": round(traj_angle, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

        return violations