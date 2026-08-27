"""
association.py
Stage 2 — Spatial association between persons and motorcycles.
Uses vertical-axis IoU which is more robust than full IoU for
riders who are seated on/above a motorcycle bbox.
"""
from __future__ import annotations
import numpy as np


def box_to_xyxy(box) -> tuple[int, int, int, int]:
    arr = box.xyxy[0].cpu().numpy().astype(int)
    return int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])


def vertical_overlap_ratio(boxA: tuple, boxB: tuple) -> float:
    """
    Fraction of boxA's vertical span that overlaps with boxB.
    boxA = person, boxB = motorcycle.
    A rider's lower body should vertically overlap the motorcycle.
    """
    _, ay1, _, ay2 = boxA
    _, by1, _, by2 = boxB
    inter_y1 = max(ay1, by1)
    inter_y2 = min(ay2, by2)
    inter_h  = max(0, inter_y2 - inter_y1)
    a_h      = max(1, ay2 - ay1)
    return inter_h / a_h


def horizontal_overlap_ratio(boxA: tuple, boxB: tuple) -> float:
    """Fraction of boxA's horizontal span that overlaps with boxB."""
    ax1, _, ax2, _ = boxA
    bx1, _, bx2, _ = boxB
    inter_x1 = max(ax1, bx1)
    inter_x2 = min(ax2, bx2)
    inter_w  = max(0, inter_x2 - inter_x1)
    a_w      = max(1, ax2 - ax1)
    return inter_w / a_w


def associate_riders(
    person_boxes: list[tuple],
    moto_boxes: list[tuple],
    v_threshold: float = 0.25,
    h_threshold: float = 0.10,
) -> dict[int, list[int]]:
    """
    Returns {moto_idx: [person_idx, ...]} grouping.
    A person is assigned to a motorcycle when both their vertical AND
    horizontal spans overlap sufficiently with that motorcycle's bbox.
    """
    groups: dict[int, list[int]] = {i: [] for i in range(len(moto_boxes))}

    for p_idx, p_box in enumerate(person_boxes):
        best_moto = -1
        best_score = 0.0
        for m_idx, m_box in enumerate(moto_boxes):
            v_ratio = vertical_overlap_ratio(p_box, m_box)
            h_ratio = horizontal_overlap_ratio(p_box, m_box)
            score   = v_ratio * h_ratio
            if v_ratio >= v_threshold and h_ratio >= h_threshold and score > best_score:
                best_score = score
                best_moto  = m_idx
        if best_moto >= 0:
            groups[best_moto].append(p_idx)

    return groups


def crop_box(frame: np.ndarray, box: tuple, pad_frac: float = 0.0) -> np.ndarray:
    """Return a cropped region of frame with optional padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad_frac)
    pad_y = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def upper_body_crop(frame: np.ndarray, person_box: tuple) -> np.ndarray:
    """Crop the upper 55% of a person bbox — where the head/helmet is."""
    x1, y1, x2, y2 = person_box
    mid_y = y1 + int((y2 - y1) * 0.55)
    return crop_box(frame, (x1, y1, x2, mid_y), pad_frac=0.05)
