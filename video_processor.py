import csv
import cv2
import numpy as np
import supervision as sv
import yaml
import os
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class VideoProcessor(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_count_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, video_path, polygon_points):
        super().__init__()
        self.video_path = video_path
        self.polygon_points = polygon_points
        self.running = True

        # --- СОЗДАЕМ КАСТОМНЫЙ МОЗГ ДЛЯ ТРЕКЕРА ---
        # Мы программно создаем файл настроек с "длинной памятью"
        self.custom_tracker_path = "custom_botsort.yaml"
        self.create_custom_tracker_config()

    def create_custom_tracker_config(self):
        """Создает файл настроек трекера с агрессивной памятью против перекрытий"""
        config = {
            'tracker_type': 'botsort',
            'track_high_thresh': 0.3,  # Порог уверенности для захвата человека
            'track_low_thresh': 0.1,  # Порог для удержания "спрятавшегося" человека
            'new_track_thresh': 0.6,  # Делаем ВЫШЕ: трекеру будет сложнее выдать новый ID!
            'track_buffer': 150,  # САМОЕ ВАЖНОЕ: Память 150 кадров (около 5 секунд) вместо 30!
            'match_thresh': 0.9,  # Позволяем связывать рамки, даже если они немного изменились
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.25,
            'with_reid': True,  # Оставляем классический BoT-SORT без тяжелой доп. модели
            'fuse_score': True,
            'model': 'yolo26n-cls.pt'
        }
        with open(self.custom_tracker_path, 'w') as f:
            yaml.dump(config, f)

    def run(self):
        model = YOLO("yolo26n.pt")

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

        polygon = np.array(self.polygon_points, dtype=np.int32)

        zone_mask = np.zeros((720, 1280), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [polygon], 255)

        cap = cv2.VideoCapture(self.video_path)

        csv_file = open('people_statistics.csv', mode='w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Время (сек)', 'Людей в зоне'])

        unique_people_in_zone = set()

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))

            # Подключаем наш кастомный файл с длинной памятью
            results = model.track(
                frame,
                classes=[0],
                conf=0.35,
                imgsz=1280,
                tracker=self.custom_tracker_path,  # ИСПОЛЬЗУЕМ СВОЙ КОНФИГ
                persist=True,
                verbose=False
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            # Фильтрация по маске
            is_inside = []
            for xyxy in detections.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1280, x2), min(720, y2)

                if x2 > x1 and y2 > y1:
                    overlap_area = cv2.countNonZero(zone_mask[y1:y2, x1:x2])
                    is_inside.append(overlap_area > 0)
                else:
                    is_inside.append(False)

            is_inside_array = np.array(is_inside, dtype=bool)
            detections_in_zone = detections[is_inside_array]

            labels = []
            current_people_in_zone = 0
            if detections_in_zone.tracker_id is not None:
                current_people_in_zone = len(detections_in_zone.tracker_id)

                for tracker_id in detections_in_zone.tracker_id:
                    unique_people_in_zone.add(tracker_id)
                    labels.append(f"#{tracker_id}")

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections_in_zone)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_in_zone,
                                                       labels=labels)

            cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)

            cv2.putText(
                annotated_frame,
                f"{current_people_in_zone}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            csv_writer.writerow([round(current_time, 2), current_people_in_zone])

            self.change_pixmap_signal.emit(annotated_frame)
            self.update_count_signal.emit(len(unique_people_in_zone))

        cap.release()

        # Удаляем временный конфиг за собой
        if os.path.exists(self.custom_tracker_path):
            os.remove(self.custom_tracker_path)

        csv_file.close()

        self.finished_signal.emit()

    def stop(self):
        self.running = False
        self.wait()