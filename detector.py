import cv2
import numpy as np
from ultralytics import YOLO
import os

class PotholeDetector:
    def __init__(self, model_path="models/pothole_best.pt"):
        self.use_custom_model = os.path.exists(model_path)
        if self.use_custom_model:
            self.model = YOLO(model_path)
        else:
            # Default COCO model sirf vehicles/objects filter karne ke liye use hoga
            self.model = YOLO("yolov8n.pt")

    def calculate_severity(self, box_area, frame_area):
        ratio = (box_area / frame_area) * 100
        # OpenCV BGR: Red = Deep/High, Yellow = Medium, Green = Little/Low
        if ratio < 1.2:
            return "Low", (0, 255, 0)         # Green
        elif ratio < 3.8:
            return "Medium", (0, 255, 255)    # Yellow
        else:
            return "High", (0, 0, 255)        # Red

    def draw_styled_box(self, img, x1, y1, x2, y2, label, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        banner_y1 = max(y1 - text_h - 8, 0)
        banner_y2 = max(y1, text_h + 8)
        cv2.rectangle(img, (x1, banner_y1), (x1 + text_w + 6, banner_y2), color, -1)

        text_color = (0, 0, 0) if color in [(0, 255, 255), (0, 255, 0)] else (255, 255, 255)
        cv2.putText(img, label, (x1 + 3, banner_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    def detect_road_potholes_cv(self, frame, vehicle_boxes, conf_thresh=0.35):
        """
        Jab custom pothole_best.pt na ho, toh cars/vehicles ko mask karke
        road surface ke dark/water-filled potholes detect karta hai.
        """
        h, w, _ = frame.shape
        frame_area = h * w

        # Sirf road region focus karein (top 12% sky/horizon ignore)
        roi_top = int(h * 0.12)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Adaptive thresholding + Canny edge texture se pothole depression/puddles pakdein
        edges = cv2.Canny(blurred, 45, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Bahut chhote noise ya poori screen jitne bade area ko filter karein
            if area < frame_area * 0.004 or area > frame_area * 0.25:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if y < roi_top:
                continue

            aspect_ratio = bw / float(bh)
            if aspect_ratio < 0.35 or aspect_ratio > 4.5:
                continue

            x1, y1, x2, y2 = x, y, x + bw, y + bh

            # Check karein ki ye box kisi Car/Vehicle ke upar toh nahi hai
            is_vehicle = False
            for (vx1, vy1, vx2, vy2) in vehicle_boxes:
                inter_x1 = max(x1, vx1)
                inter_y1 = max(y1, vy1)
                inter_x2 = min(x2, vx2)
                inter_y2 = min(y2, vy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    if inter_area / float(bw * bh) > 0.25:
                        is_vehicle = True
                        break

            if not is_vehicle:
                # Pseudo confidence based on contour solidity
                hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1
                solidity = area / hull_area
                if solidity >= conf_thresh:
                    boxes.append((x1, y1, x2, y2, min(round(solidity, 2), 0.96)))

        return boxes

    def get_vehicle_masks(self, frame):
        """COCO model se Cars, Trucks, Buses, Bikes ki location nikalta hai taaki unhe ignore kiya ja sake"""
        results = self.model.predict(source=frame, conf=0.30, verbose=False)[0]
        vehicle_boxes = []
        # COCO IDs: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_classes = {1, 2, 3, 5, 7}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                vehicle_boxes.append((vx1, vy1, vx2, vy2))
        return vehicle_boxes

    def resize_frame(self, frame, target_width=640):
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame
        scale = target_width / float(w)
        return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def process_image(self, image_path, output_path, conf_thresh=0.35):
        img = cv2.imread(image_path)
        img = self.resize_frame(img, target_width=720)
        h, w, _ = img.shape
        frame_area = h * w
        detections = []

        if self.use_custom_model:
            results = self.model.predict(source=img, conf=conf_thresh, verbose=False)[0]
            raw_boxes = [(*map(int, b.xyxy[0]), float(b.conf[0])) for b in results.boxes]
        else:
            vehicles = self.get_vehicle_masks(img)
            raw_boxes = self.detect_road_potholes_cv(img, vehicles, conf_thresh)

        for idx, (x1, y1, x2, y2, conf) in enumerate(raw_boxes, start=1):
            box_area = (x2 - x1) * (y2 - y1)
            severity, color = self.calculate_severity(box_area, frame_area)
            est_area_sqm = round((box_area / frame_area) * 12.5, 2)

            detections.append({
                "id": idx,
                "severity": severity,
                "confidence": round(conf * 100, 1),
                "area_px": box_area,
                "est_area_sqm": est_area_sqm,
                "frame": 1
            })

            label = f"#{idx} {severity.upper()} ({int(conf*100)}%)"
            self.draw_styled_box(img, x1, y1, x2, y2, label, color)

        cv2.imwrite(output_path, img)
        return output_path, detections

    def process_video(self, video_path, output_path, conf_thresh=0.35, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Compact video size (width = 640px)
        target_w = min(orig_w, 640)
        target_h = int(orig_h * (target_w / float(orig_w)))
        frame_area = target_w * target_h

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        unique_potholes = {}
        next_id = 1
        active_tracks = []  # [(id, cx, cy, last_frame)]
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if self.use_custom_model:
                results = self.model.track(
                    source=frame, conf=conf_thresh, persist=True, tracker="bytetrack.yaml", verbose=False
                )[0]
                current_boxes = []
                if results.boxes is not None and results.boxes.id is not None:
                    for box, tid, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                              results.boxes.id.int().cpu().tolist(),
                                              results.boxes.conf.cpu().tolist()):
                        x1, y1, x2, y2 = map(int, box)
                        current_boxes.append((tid, x1, y1, x2, y2, conf))
            else:
                vehicles = self.get_vehicle_masks(frame)
                raw_boxes = self.detect_road_potholes_cv(frame, vehicles, conf_thresh)
                current_boxes = []

                # Simple centroid tracker taaki ek hi pothole baar-baar count na ho
                updated_tracks = []
                for (x1, y1, x2, y2, conf) in raw_boxes:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    matched_id = None
                    for (tid, px, py, last_f) in active_tracks:
                        if frame_num - last_f <= 12 and np.hypot(cx - px, cy - py) < 75:
                            matched_id = tid
                            break
                    if matched_id is None:
                        matched_id = next_id
                        next_id += 1
                    updated_tracks.append((matched_id, cx, cy, frame_num))
                    current_boxes.append((matched_id, x1, y1, x2, y2, conf))

                # Purane tracks maintain karein
                for t in active_tracks:
                    if frame_num - t[3] <= 12 and all(t[0] != u[0] for u in updated_tracks):
                        updated_tracks.append(t)
                active_tracks = updated_tracks

            for (track_id, x1, y1, x2, y2, conf) in current_boxes:
                box_area = (x2 - x1) * (y2 - y1)
                severity, color = self.calculate_severity(box_area, frame_area)
                est_area_sqm = round((box_area / frame_area) * 12.5, 2)

                if track_id not in unique_potholes or box_area > unique_potholes[track_id]["area_px"]:
                    unique_potholes[track_id] = {
                        "id": track_id,
                        "severity": severity,
                        "confidence": round(conf * 100, 1),
                        "area_px": box_area,
                        "est_area_sqm": est_area_sqm,
                        "frame": frame_num
                    }

                label = f"ID:{track_id} | {severity.upper()}"
                self.draw_styled_box(frame, x1, y1, x2, y2, label, color)

            if progress_callback and frame_num % 4 == 0:
                progress_callback(frame, frame_num, total_frames, len(unique_potholes))

            out.write(frame)

        cap.release()
        out.release()
        return output_path, list(unique_potholes.values())