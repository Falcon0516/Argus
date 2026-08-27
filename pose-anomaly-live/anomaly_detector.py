import numpy as np
from collections import deque
import hashlib
import time

class AnomalyDetector:
    def __init__(self, sequence_len: int = 30):
        self.seq_len = sequence_len
        self.kpt_buffers = {}       # track_id -> deque of (17, 2) normalized keypoints
        self.ratio_buffers = {}     # track_id -> deque of bbox aspect ratios (w / h)
        self.torso_y_buffers = {}   # track_id -> deque of absolute torso y-coords
        self.timestamps = {}

    def update(self, track_id: int, norm_kpts: np.ndarray, bbox: np.ndarray, raw_kpts: np.ndarray) -> dict:
        """
        Updates the temporal rolling buffer and computes kinematic anomaly scores.
        """
        current_time = time.time()
        
        if track_id not in self.kpt_buffers:
            self.kpt_buffers[track_id] = deque(maxlen=self.seq_len)
            self.ratio_buffers[track_id] = deque(maxlen=self.seq_len)
            self.torso_y_buffers[track_id] = deque(maxlen=self.seq_len)
            self.timestamps[track_id] = deque(maxlen=self.seq_len)

        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        aspect_ratio = width / height

        # Compute mid-torso y-level (shoulders and hips midpoint)
        torso_y = np.mean([raw_kpts[5, 1], raw_kpts[6, 1], raw_kpts[11, 1], raw_kpts[12, 1]])

        self.kpt_buffers[track_id].append(norm_kpts)
        self.ratio_buffers[track_id].append(aspect_ratio)
        self.torso_y_buffers[track_id].append(torso_y)
        self.timestamps[track_id].append(current_time)

        status = "NORMAL"
        confidence = 0.0

        # Run action analysis once we have at least 12 frames of tracking data
        if len(self.kpt_buffers[track_id]) >= 12:
            # 1. FALL DETECTION
            torso_delta = self.torso_y_buffers[track_id][-1] - self.torso_y_buffers[track_id][0]
            current_ratio = self.ratio_buffers[track_id][-1]

            if current_ratio > 1.05 and torso_delta > 35:
                status = "FALL_DETECTED"
                confidence = min(0.96, (current_ratio * 0.4) + (torso_delta / 80.0) * 0.5)

            # 2. FIGHT / AGGRESSIVE STANCE
            elif self._check_fight(self.kpt_buffers[track_id]):
                status = "FIGHT_STANCE"
                confidence = 0.84

            # 3. SUSPICIOUS / ABNORMAL GAIT
            elif self._check_suspicious_gait(self.kpt_buffers[track_id]):
                status = "SUSPICIOUS_GAIT"
                confidence = 0.76

        sec65b_hash = self.generate_sec65b_hash(track_id, status, current_time)

        return {
            "track_id": track_id,
            "status": status,
            "confidence": round(confidence, 2),
            "evidence_hash": sec65b_hash
        }

    def _check_fight(self, buffer: deque) -> bool:
        """Flags rapid hand velocity paired with an elevated defensive/strike posture."""
        recent = np.array(buffer) # Shape (T, 17, 2)
        l_wrist_y = recent[-1, 9, 1]
        r_wrist_y = recent[-1, 10, 1]
        
        hands_elevated = (l_wrist_y < 0.15 or r_wrist_y < 0.15)

        if len(recent) >= 8:
            wrist_movement = np.abs(np.diff(recent[-8:, [9, 10], :], axis=0)).sum()
            if hands_elevated and wrist_movement > 0.95:
                return True
        return False

    def _check_suspicious_gait(self, buffer: deque) -> bool:
        """Flags abnormal stance staggering or sudden lateral lurching."""
        recent = np.array(buffer)
        if len(recent) >= 12:
            lateral_sway = np.std(recent[:, [15, 16], 0])
            if lateral_sway > 0.28:
                return True
        return False

    def generate_sec65b_hash(self, track_id: int, status: str, timestamp: float) -> str:
        """Creates an immutable SHA-256 evidence string for legal admissibility."""
        payload = f"ARGUS-POSE|ID:{track_id}|ACTION:{status}|TIME:{timestamp}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()