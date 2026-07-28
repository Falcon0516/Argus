"""
Perspective correction for cropped license plate images.

fast-alpr's YOLO detector returns an axis-aligned bounding box, so a plate
photographed at a steep angle comes out as a skewed/keystoned crop, which
hurts OCR accuracy. This module tries to find the plate's actual quadrilateral
within that crop (via edge detection + contour analysis) and warps it to a
straight-on rectangle before OCR sees it.

If no clean quadrilateral is found (e.g. the plate is already near-frontal,
or edges are too noisy), it safely falls back to returning the original crop
unchanged - so this never makes things worse, only better on steep angles.
"""

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]        # bottom-right has largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]     # bottom-left has largest y-x
    return rect


def correct_perspective(
    cropped_plate: np.ndarray,
    out_width: int = 240,
    out_height: int = 80,
) -> np.ndarray:
    """
    Attempt to find the plate's quadrilateral inside a (possibly angled)
    detection crop and warp it to a straight-on rectangle. Falls back to
    the original crop if no good quadrilateral is found.
    """
    if cropped_plate is None or cropped_plate.size == 0:
        return cropped_plate

    h, w = cropped_plate.shape[:2]
    if h < 10 or w < 10:
        return cropped_plate

    gray = (
        cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        if cropped_plate.ndim == 3
        else cropped_plate
    )
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 30, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cropped_plate

    plate_area = h * w
    best_quad = None
    best_area = 0.0

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        # Plate quad should cover a meaningful chunk of the crop (not tiny
        # noise) but not the whole frame border either.
        if 0.3 * plate_area < area < 0.98 * plate_area and area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return cropped_plate

    pts = best_quad.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    dst = np.array(
        [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(cropped_plate, matrix, (out_width, out_height))
