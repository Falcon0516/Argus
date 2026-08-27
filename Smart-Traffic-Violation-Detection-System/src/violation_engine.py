"""
violation_engine.py
Stage 3 — Per-motorcycle violation rules.
Input:  grouped riders, per-rider helmet status, plate detections.
Output: list of Violation dicts ready for logging and drawing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

TRIPLE_RIDING_MIN = 3   # riders per motorcycle to flag

# Helmet model class IDs (confirmed from model inspection)
HELMET_CLASS_ID = 1   # wearing helmet  → compliant
HEAD_CLASS_ID   = 0   # bare head       → violation


@dataclass
class Violation:
    violation_type: str          # "TRIPLE_RIDING" | "NO_HELMET" | "PLATE"
    severity: str                # "HIGH" | "MEDIUM"
    motorcycle_idx: int
    rider_idx: int | None        # None for motorcycle-level violations
    rider_count: int = 0
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    conf: float = 0.0

    def log_line(self) -> str:
        if self.violation_type == "TRIPLE_RIDING":
            return f"[{self.timestamp}] ⚠ TRIPLE RIDING — {self.rider_count} riders on motorcycle #{self.motorcycle_idx}"
        if self.violation_type == "NO_HELMET":
            return f"[{self.timestamp}] ⛑ NO HELMET — rider #{self.rider_idx} on motorcycle #{self.motorcycle_idx} (conf {self.conf:.2f})"
        if self.violation_type == "PLATE":
            return f"[{self.timestamp}] 🔢 PLATE DETECTED — motorcycle #{self.motorcycle_idx} ({self.label} conf {self.conf:.2f})"
        return f"[{self.timestamp}] {self.violation_type}"


class ViolationEngine:

    def __init__(self, triple_min: int = TRIPLE_RIDING_MIN):
        self.triple_min = triple_min

    def evaluate(
        self,
        groups: dict[int, list[int]],
        helmet_statuses: dict[int, dict],   # {person_idx: {"has_helmet": bool, "conf": float}}
        plate_detections: dict[int, list],  # {moto_idx: [{"label": str, "conf": float}]}
    ) -> list[Violation]:
        violations: list[Violation] = []

        for mc_idx, rider_idxs in groups.items():
            if not rider_idxs:
                continue

            # ── Triple riding ─────────────────────────────────────────────
            if len(rider_idxs) >= self.triple_min:
                violations.append(Violation(
                    violation_type="TRIPLE_RIDING",
                    severity="HIGH",
                    motorcycle_idx=mc_idx,
                    rider_idx=None,
                    rider_count=len(rider_idxs),
                    label=f"⚠ TRIPLE RIDING ({len(rider_idxs)} riders)",
                ))

            # ── No helmet per rider ───────────────────────────────────────
            for r_idx in rider_idxs:
                status = helmet_statuses.get(r_idx)
                if status and not status["has_helmet"]:
                    violations.append(Violation(
                        violation_type="NO_HELMET",
                        severity="HIGH",
                        motorcycle_idx=mc_idx,
                        rider_idx=r_idx,
                        label="⛑ NO HELMET",
                        conf=status["conf"],
                    ))

            # ── Plate detections ─────────────────────────────────────────
            for plate in plate_detections.get(mc_idx, []):
                violations.append(Violation(
                    violation_type="PLATE",
                    severity="MEDIUM",
                    motorcycle_idx=mc_idx,
                    rider_idx=None,
                    label=plate["label"],
                    conf=plate["conf"],
                ))

        return violations


def classify_helmet_result(helmet_result) -> dict:
    """
    Parse a helmet model result on a rider crop.
    Returns {"has_helmet": bool, "conf": float}.
    Helmet class=1 means compliant; head class=0 means no helmet.
    """
    if helmet_result is None or helmet_result.boxes is None or len(helmet_result.boxes) == 0:
        return {"has_helmet": True, "conf": 0.0}  # no detection → don't flag

    # Pick the highest-confidence detection
    best_conf  = 0.0
    best_class = -1
    for box in helmet_result.boxes:
        c = float(box.conf[0])
        if c > best_conf:
            best_conf  = c
            best_class = int(box.cls[0])

    has_helmet = (best_class == HELMET_CLASS_ID)
    return {"has_helmet": has_helmet, "conf": best_conf}
