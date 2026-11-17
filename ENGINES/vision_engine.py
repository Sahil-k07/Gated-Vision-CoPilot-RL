import cv2
import numpy as np


class VisionEngine:
    """
    Heavyweight Vision Perception Engine.
    Accurate obstacle distance and visual risk estimation, but high compute cost.
    """
    COMPUTE_COST_MS = 18.5

    def __init__(self):
        self.last_keypoints = None
        self.call_count = 0

    def process_frame(self, frame: np.ndarray, gt_hint: float = None) -> dict:
        """
        Executes heavy perception on the current RGB frame.
        Returns estimated obstacle distance, visual clutter score, and feature map.
        """
        self.call_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute visual complexity via Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.hypot(grad_x, grad_y)) / 255.0)

        # Extract ROI in center lane to detect lead vehicle / obstacle size
        roi = frame[75:170, 90:230]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_area = max([cv2.contourArea(c) for c in contours], default=50.0)
        raw_est_dist = np.clip(1800.0 / (np.sqrt(detected_area) + 15.0), 5.0, 65.0)

        # Fuse with sensor model with very low sensor noise (high accuracy)
        if gt_hint is not None:
            est_dist = 0.92 * gt_hint + 0.08 * raw_est_dist + np.random.normal(0, 0.35)
        else:
            est_dist = raw_est_dist

        est_dist = float(np.clip(est_dist, 5.0, 70.0))

        return {
            "estimated_distance_m": est_dist,
            "visual_complexity": edge_energy,
            "compute_time_ms": self.COMPUTE_COST_MS,
            "uncertainty": 0.02,
            "mode": "FULL_VISION"
        }