import cv2
import numpy as np

# 17 Standard COCO Keypoint Definitions
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Limb connection pairs for skeleton rendering
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial landmarks
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Shoulders and arms
    (5, 11), (6, 12), (11, 12),               # Torso / Spine
    (11, 13), (13, 15), (12, 14), (14, 16)    # Hips and legs
]

class PoseProcessor:
    def __init__(self, conf_threshold: float = 0.35):
        self.conf_thresh = conf_threshold

    def normalize_keypoints(self, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Normalizes (x, y) coordinates relative to the person's bounding box center and size.
        Ensures temporal sequence invariance across different camera distances.
        """
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        normalized = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf >= self.conf_thresh:
                normalized[i, 0] = (x - cx) / bw
                normalized[i, 1] = (y - cy) / bh
            else:
                normalized[i, 0] = 0.0
                normalized[i, 1] = 0.0
        return normalized

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color=(0, 255, 245)):
        """Draws skeletal links and joint circles."""
        # Draw limb connections
        for p1, p2 in SKELETON_PAIRS:
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            if c1 >= self.conf_thresh and c2 >= self.conf_thresh:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)

        # Draw joint nodes
        for x, y, conf in keypoints:
            if conf >= self.conf_thresh:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 165, 255), -1, cv2.LINE_AA)