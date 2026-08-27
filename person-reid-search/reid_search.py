import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from reid_core import ReidFeatureExtractor

class ArgusReIDSearch:
    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x0_25'):
        self.detector = YOLO(yolo_model)
        self.extractor = ReidFeatureExtractor(model_name=reid_model)
        self.gallery = []  # Stores: {'camera_id', 'timestamp', 'bbox', 'crop', 'embedding'}

    def process_frame(self, frame, camera_id="CAM_01", timestamp="00:00:00"):
        """Detects persons, crops them, and appends embeddings to the gallery."""
        results = self.detector(frame, verbose=False)[0]
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Class 0 corresponds to 'person' in COCO dataset
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0 and (x2 - x1) > 20 and (y2 - y1) > 40:
                    embedding = self.extractor.extract(crop)
                    self.gallery.append({
                        "camera_id": camera_id,
                        "timestamp": timestamp,
                        "bbox": (x1, y1, x2, y2),
                        "crop": crop,
                        "embedding": embedding
                    })

    def search_target(self, query_img_path, top_k=5, sim_threshold=0.6):
        """Matches a target image against all recorded sightings in the gallery."""
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            raise FileNotFoundError(f"Query image not found at {query_img_path}")

        query_emb = self.extractor.extract(query_img)
        matches = []

        for record in self.gallery:
            # Cosine similarity: 1 - cosine_distance
            sim = 1.0 - cosine(query_emb, record["embedding"])
            if sim >= sim_threshold:
                matches.append({
                    "similarity": round(float(sim), 4),
                    "camera_id": record["camera_id"],
                    "timestamp": record["timestamp"],
                    "crop": record["crop"]
                })

        # Rank candidate matches from highest to lowest similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

# ==================== DEMO USAGE ====================
if __name__ == "__main__":
    reid_system = ArgusReIDSearch()

    # 1. Simulate reading video feed to populate the gallery
    video_path = "test_feed.mp4" # Replace with your video file or 0 for webcam
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    print("[INFO] Building gallery from camera stream...")
    while cap.isOpened() and frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break
        reid_system.process_frame(frame, camera_id="NODE_01", timestamp=f"frame_{frame_idx}")
        frame_idx += 1
    cap.release()

    # 2. Search for the missing person / person-of-interest
    query_image_path = "target_person.jpg"
    print(f"[INFO] Querying target: {query_image_path}...")
    results = reid_system.search_target(query_image_path, top_k=3, sim_threshold=0.5)

    for i, res in enumerate(results, 1):
        print(f"Match #{i}: Camera: {res['camera_id']} | Time: {res['timestamp']} | Confidence: {res['similarity'] * 100:.2f}%")
        cv2.imshow(f"Rank_{i}_Match", res["crop"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()