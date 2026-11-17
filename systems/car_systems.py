import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ENGINES.vision_engine import VisionEngine
from ENGINES.frame_interpolator import FrameInterpolator
from systems.data_feeder import DataFeeder


class CarSystems:
    """Computes vehicle safety metrics and Time-To-Collision (TTC)."""

    @staticmethod
    def compute_risk_score(speed_kmh: float, steering_deg: float, est_dist_m: float) -> float:
        speed_ms = max(speed_kmh / 3.6, 0.1)
        ttc = est_dist_m / speed_ms
        steering_factor = 1.0 + abs(steering_deg) / 25.0
        raw_risk = (2.5 / max(ttc, 0.2)) * steering_factor
        return float(np.clip(raw_risk, 0.0, 1.0))


class GatedVisionRLEnv(gym.Env):
    """
    Reinforcement Learning MDP Environment for Gated-Vision-CoPilot.
    
    State Space (7D Continuous):
        0: Normalized Speed (speed_kmh / 120.0)
        1: Normalized Steering Angle (abs(steering) / 45.0)
        2: Normalized Acceleration (accel / 5.0)
        3: Normalized Estimated Obstacle Distance (est_dist / 70.0)
        4: Normalized Steps Since Last Vision Run (skips / 15.0)
        5: Epistemic Uncertainty [0, 1]
        6: Kinematic Risk Score [0, 1]

    Action Space (2 Discrete):
        0: GATE CLOSED -> Use fast FrameInterpolator (saves compute)
        1: GATE OPEN   -> Trigger full VisionEngine (high accuracy, high cost)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_dir="dataset/driving_sequence", max_episode_steps=250):
        super().__init__()
        self.feeder = DataFeeder(dataset_dir=dataset_dir)
        self.vision_engine = VisionEngine()
        self.interpolator = FrameInterpolator()
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.current_step = 0
        self.current_frame = None
        self.current_telemetry = None
        self.last_perception = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.vision_engine.call_count = 0
        self.interpolator.call_count = 0

        random_start = options.get("random_start", True) if options else True
        step_data = self.feeder.reset(random_start=random_start)
        self.current_frame, self.current_telemetry = step_data

        # Initialize step 0 with an anchor vision pass
        self.last_perception = self.vision_engine.process_frame(
            self.current_frame, gt_hint=self.current_telemetry["gt_obstacle_dist_m"]
        )
        self.interpolator.update_anchor(
            self.current_frame, self.last_perception["estimated_distance_m"]
        )

        obs = self._build_observation()
        return obs, {}

    def step(self, action: int):
        self.current_step += 1
        action = int(action)

        gt_dist = self.current_telemetry["gt_obstacle_dist_m"]
        hazard = self.current_telemetry["hazard_event"]

        # Execute selected perception pathway
        if action == 1:
            perception = self.vision_engine.process_frame(self.current_frame, gt_hint=gt_dist)
            self.interpolator.update_anchor(self.current_frame, perception["estimated_distance_m"])
        else:
            perception = self.interpolator.interpolate(
                self.current_frame,
                self.current_telemetry["speed_kmh"],
                self.current_telemetry["acceleration_ms2"]
            )

        self.last_perception = perception
        est_dist = perception["estimated_distance_m"]
        tracking_error = abs(est_dist - gt_dist)

        # --- Multi-Objective RL Reward Function ---
        # 1. Accuracy term: penalize perception drift from ground truth
        accuracy_reward = 1.5 * np.exp(-0.25 * tracking_error)

        # 2. Compute efficiency term: reward skipping heavy vision when safe
        compute_penalty = -0.35 if action == 1 else +0.25

        # 3. Safety-critical penalty: heavy penalty if skipping vision during hazard or high drift
        safety_penalty = 0.0
        if hazard == 1 and action == 0 and self.interpolator.consecutive_skips > 1:
            safety_penalty -= 2.0
        if tracking_error > 6.0:
            safety_penalty -= 1.5

        reward = float(accuracy_reward + compute_penalty + safety_penalty)

        # Advance feeder to next timestep
        next_data = self.feeder.get_step_data()
        terminated = next_data is None
        truncated = self.current_step >= self.max_episode_steps

        if not terminated:
            self.current_frame, self.current_telemetry = next_data

        obs = self._build_observation()
        info = {
            "action_gate": action,
            "tracking_error_m": tracking_error,
            "gt_dist_m": gt_dist,
            "est_dist_m": est_dist,
            "compute_ms": perception["compute_time_ms"],
            "hazard": hazard,
            "uncertainty": perception["uncertainty"]
        }
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        t = self.current_telemetry
        p = self.last_perception
        risk = CarSystems.compute_risk_score(
            t["speed_kmh"], t["steering_angle_deg"], p["estimated_distance_m"]
        )
        obs = np.array([
            np.clip(t["speed_kmh"] / 120.0, 0.0, 1.0),
            np.clip(abs(t["steering_angle_deg"]) / 45.0, 0.0, 1.0),
            np.clip(t["acceleration_ms2"] / 5.0, -1.0, 1.0),
            np.clip(p["estimated_distance_m"] / 70.0, 0.0, 1.0),
            np.clip(self.interpolator.consecutive_skips / 15.0, 0.0, 1.0),
            np.clip(p["uncertainty"], 0.0, 1.0),
            risk
        ], dtype=np.float32)
        return obs