import cv2
import numpy as np
from ultralytics import YOLO

class PotholeSegmentor:
    def __init__(self, model_path="model/best.pt", conf_thresh=0.35, px_to_m2_factor=0.00015):
        """
        px_to_m2_factor: Camera calibration factor converting mask pixel count to square meters.
        Adjust based on camera height and angle.
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.calibration_factor = px_to_m2_factor

    def classify_severity(self, area_m2):
        if area_m2 < 0.05:
            return "Minor"
        elif area_m2 < 0.20:
            return "Moderate"
        else:
            return "Severe"

    def process_frame(self, frame):
        """
        Runs segmentation, overlays colored masks, and returns metrics.
        """
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)[0]
        detections = []
        annotated_frame = frame.copy()

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.cpu().numpy()
            
            h, w = frame.shape[:2]

            for idx, mask in enumerate(masks):
                # Resize mask to frame dimensions
                resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask = (resized_mask > 0.5).astype(np.uint8)

                # Calculate area
                mask_pixel_area = np.sum(binary_mask)
                real_area_m2 = round(float(mask_pixel_area * self.calibration_factor), 3)
                severity = self.classify_severity(real_area_m2)
                conf = float(boxes[idx].conf[0])

                # Severity-based color coding (BGR format)
                color = (0, 255, 0) if severity == "Minor" else (0, 165, 255) if severity == "Moderate" else (0, 0, 255)

                # Overlay mask on frame
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[binary_mask == 1] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.4, 0)

                # Draw contour boundary
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, color, 2)

                # Label text
                x1, y1, x2, y2 = map(int, boxes[idx].xyxy[0])
                label = f"Pothole: {severity} ({real_area_m2} m2)"
                cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({
                    "damage_type": "pothole",
                    "severity": severity,
                    "area_m2": real_area_m2,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        return annotated_frame, detections