"""
visualizer.py
Stage 4 — Draw all annotations onto the frame with vivid,
semantically-colored bounding boxes and violation banners.
"""
from __future__ import annotations
import cv2
import numpy as np
from .violation_engine import Violation

# Color palette (BGR)
COLOR = {
    "motorcycle":      (0,   220, 255),   # Cyan
    "person_ok":       (50,  220,  50),   # Green  — helmet on, not triple
    "person_violation":(30,   30, 255),   # Red    — no helmet OR triple riding
    "person_unknown":  (180, 180, 180),   # Grey   — not on a motorcycle
    "plate":           (0,   200, 255),   # Gold
    "banner_triple":   (0,    0,  220),   # Red banner
    "banner_nohelmet": (0,   90,  255),   # Orange-red banner
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = 0.50
FONT_MED   = 0.60
THICK      = 2


def _label_box(frame, x1, y1, x2, y2, label, color, font_scale=FONT_SMALL):
    """Draw a filled label tab + border box."""
    (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, 1)
    tab_y1 = max(0, y1 - th - 8)
    tab_y2 = y1
    cv2.rectangle(frame, (x1, tab_y1), (x1 + tw + 8, tab_y2), color, -1)
    cv2.rectangle(frame, (x1, y1),     (x2, y2),               color, THICK)
    cv2.putText(frame, label, (x1 + 4, tab_y2 - 3),
                FONT, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 4, tab_y2 - 3),
                FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _violation_banner(frame, x1, y1, x2, text, bg_color):
    """Draw a wide violation banner above a bounding box."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_MED, 1)
    bx1 = x1
    bx2 = max(x2, x1 + tw + 16)
    by1 = max(0, y1 - th - 14)
    by2 = y1 - 2
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.putText(frame, text, (bx1 + 8, by2 - 4),
                FONT, FONT_MED, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + 8, by2 - 4),
                FONT, FONT_MED, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    person_boxes:  list[tuple],
    moto_boxes:    list[tuple],
    groups:        dict[int, list[int]],
    helmet_statuses: dict[int, dict],
    violations:    list[Violation],
    plate_texts:   dict[int, str],   # {moto_idx: plate_label}
) -> np.ndarray:
    """Annotate frame in-place. Returns the frame."""

    # Build per-person and per-motorcycle violation sets for color lookup
    violation_persons = set()
    violation_motos   = set()
    triple_motos      = set()
    nohelmet_motos    = set()

    for v in violations:
        if v.rider_idx is not None:
            violation_persons.add(v.rider_idx)
        if v.violation_type in ("TRIPLE_RIDING", "NO_HELMET"):
            violation_motos.add(v.motorcycle_idx)
        if v.violation_type == "TRIPLE_RIDING":
            triple_motos.add(v.motorcycle_idx)
        if v.violation_type == "NO_HELMET":
            nohelmet_motos.add(v.motorcycle_idx)

    # All rider person indices (associated to any motorcycle)
    all_riders = set(p for riders in groups.values() for p in riders)

    # ── Draw persons ──────────────────────────────────────────────────────
    for p_idx, box in enumerate(person_boxes):
        x1, y1, x2, y2 = box
        if p_idx in violation_persons:
            color = COLOR["person_violation"]
            status = helmet_statuses.get(p_idx, {})
            label = f"NO HELMET {status.get('conf', 0):.2f}"
        elif p_idx in all_riders:
            color = COLOR["person_ok"]
            label = "rider ✓"
        else:
            color = COLOR["person_unknown"]
            label = "person"
        _label_box(frame, x1, y1, x2, y2, label, color)

    # ── Draw motorcycles ──────────────────────────────────────────────────
    for m_idx, box in enumerate(moto_boxes):
        x1, y1, x2, y2 = box
        rider_count = len(groups.get(m_idx, []))
        color = COLOR["motorcycle"]
        label = f"motorcycle #{m_idx} [{rider_count} rider{'s' if rider_count != 1 else ''}]"
        _label_box(frame, x1, y1, x2, y2, label, color)

        # Violation banners above motorcycle box
        banner_y = y1
        if m_idx in triple_motos:
            _violation_banner(frame, x1, banner_y, x2,
                              f"⚠ TRIPLE RIDING ({rider_count})", COLOR["banner_triple"])
            banner_y -= 28
        if m_idx in nohelmet_motos:
            _violation_banner(frame, x1, banner_y, x2,
                              "⛑ NO HELMET RIDER", COLOR["banner_nohelmet"])

        # Plate text inside motorcycle box bottom
        if m_idx in plate_texts:
            pt = plate_texts[m_idx]
            _label_box(frame, x1, y2 - 28, x2, y2,
                       f"🔢 {pt}", COLOR["plate"])

    return frame


def draw_hud(frame: np.ndarray, fps: float, counts: dict, n_violations: int) -> None:
    """Semi-transparent dark HUD in top-left corner."""
    lines = [
        (f"FPS: {fps:.1f}",                  (0, 220, 255)),
        (f"Motorcycles: {counts['moto']}",   (0, 220, 255)),
        (f"Riders: {counts['riders']}",      (200, 200, 200)),
        (f"Violations: {n_violations}",       (30, 30, 255) if n_violations else (100, 255, 100)),
    ]
    pad, lh = 8, 24
    box_h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, box_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    for i, (text, col) in enumerate(lines):
        y = pad + (i + 1) * lh - 4
        cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
