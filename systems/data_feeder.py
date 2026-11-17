import os
import csv
import cv2
import numpy as np


class DataFeeder:
    """Streams synchronized driving frames and CAN-bus telemetry for the RL environment."""

    def __init__(self, dataset_dir=os.path.join("dataset", "driving_sequence")):
        self.dataset_dir = dataset_dir
        self.frames_dir = os.path.join(dataset_dir, "frames")
        self.telemetry_file = os.path.join(dataset_dir, "telemetry.csv")
        self.records = []
        self.cursor = 0
        self._load_metadata()

    def _load_metadata(self):
        if not os.path.exists(self.telemetry_file):
            raise FileNotFoundError(
                f"Dataset not found at {self.telemetry_file}. Run 'python setup_dataset.py' first."
            )
        with open(self.telemetry_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append({
                    "frame_id": int(row["frame_id"]),
                    "speed_kmh": float(row["speed_kmh"]),
                    "steering_angle_deg": float(row["steering_angle_deg"]),
                    "acceleration_ms2": float(row["acceleration_ms2"]),
                    "gt_obstacle_dist_m": float(row["gt_obstacle_dist_m"]),
                    "hazard_event": int(row["hazard_event"])
                })

    def reset(self, random_start=False):
        if random_start and len(self.records) > 200:
            self.cursor = int(np.random.randint(0, len(self.records) - 150))
        else:
            self.cursor = 0
        return self.get_step_data()

    def get_step_data(self):
        if self.cursor >= len(self.records):
            return None
        rec = self.records[self.cursor]
        frame_path = os.path.join(self.frames_dir, f"frame_{rec['frame_id']:04d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.cursor += 1
        return frame, rec

    def __len__(self):
        return len(self.records)