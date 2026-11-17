import cv2
import numpy as np


class FrameInterpolator:
    """
    Lightweight Frame & State Interpolator.
    Used when the RL Gating Agent bypasses the full Vision Engine.
    """
    COMPUTE_COST_MS = 1.5

    def __init__(self):
        self.prev_gray = None
        self.cached_distance = 45.0
        self.consecutive_skips = 0
        self.call_count = 0

    def update_anchor(self, frame: np.ndarray, true_vision_dist: float):
        """Resets interpolation anchor whenever the full Vision Engine runs."""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cached_distance = true_vision_dist
        self.consecutive_skips = 0

    def interpolate(self, frame: np.ndarray, speed_kmh: float, accel_ms2: float) -> dict:
        """
        Fast approximation of perception state using frame-to-frame pixel diff
        and kinematic motion propagation.
        """
        self.call_count += 1
        self.consecutive_skips += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None:
            flow_diff = float(np.mean(cv2.absdiff(curr_gray, self.prev_gray)) / 255.0)
        else:
            flow_diff = 0.05

        self.prev_gray = curr_gray

        # Propagate obstacle distance using kinematic heuristics + drift noise
        dt = 0.05  # 20 FPS step
        velocity_ms = speed_kmh / 3.6
        drift = np.random.normal(0, 0.65 * np.sqrt(self.consecutive_skips))
        kinematic_delta = -0.15 * accel_ms2 * dt - (flow_diff - 0.04) * velocity_ms * dt * 2.5

        self.cached_distance = float(np.clip(self.cached_distance + kinematic_delta + drift, 5.0, 70.0))
        uncertainty = float(min(1.0, 0.08 * self.consecutive_skips + flow_diff * 1.5))

        return {
            "estimated_distance_m": self.cached_distance,
            "visual_complexity": flow_diff,
            "compute_time_ms": self.COMPUTE_COST_MS,
            "uncertainty": uncertainty,
            "mode": "INTERPOLATED"
        }