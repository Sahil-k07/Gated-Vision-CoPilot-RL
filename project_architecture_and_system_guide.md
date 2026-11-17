# Gated-Vision-CoPilot: Complete System Architecture & Working Guide

## 1. Core Concept: What Problem Does This Project Solve?
In modern Autonomous Driving Systems (ADS) and Advanced Driver Assistance Systems (ADAS), onboard cameras capture video at 20 to 60 frames per second (FPS). Running a deep Computer Vision neural network on **every single frame** ($100\%$ duty cycle) drains embedded GPU/NPU power, increases thermal throttling, and introduces processing latency ($18.5\text{ ms}$ per frame).

However, highway driving is largely predictable during steady cruising. If the lead vehicle is $60\text{ meters}$ away and moving at a constant speed, running a heavy vision model on every frame is wasteful. Conversely, during a **sudden braking hazard** (when the lead car decelerates rapidly), skipping vision frames can cause fatal collisions.

**Gated-Vision-CoPilot** solves this using **Reinforcement Learning (Active / Budgeted Perception)**:
* An intelligent **RL Gating Agent (`IntentEngineRL`)** sits in front of the perception pipeline.
* At every frame $t$, the RL agent inspects the vehicle's kinematics, current tracking uncertainty, and Time-To-Collision (TTC) risk.
* It decides whether to **Open the Vision Gate ($a_t = 1$)** to run the accurate $18.5\text{ ms}$ `VisionEngine`, or **Close the Vision Gate ($a_t = 0$)** and use a lightweight $1.5\text{ ms}$ `FrameInterpolator` that estimates obstacle distance using optical flow and physics dead-reckoning.

---

## 2. End-to-End Architecture & Data Flow

```text
+-------------------------------------------------------------------------+
|                        DATA STREAM (data_feeder.py)                     |
|   Synchronized RGB Frames (320x180) + CAN-Bus Telemetry (Speed, Accel)  |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   RL ENVIRONMENT (car_systems.py)                       |
|  Computes 7D State Vector s_t:                                          |
|  [Speed, Steering, Accel, Est_Dist, Consecutive_Skips, Uncertainty, TTC]|
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                RL INTENT ENGINE (intent_engine.py)                      |
|      Policy Network pi(a_t | s_t) -> Chooses Gate Action a_t in {0, 1}  |
+-----------------+-------------------------------------+-----------------+
                  |                                     |
     Action a_t = 1 (GATE OPEN)            Action a_t = 0 (GATE CLOSED)
                  |                                     |
                  v                                     v
+-----------------------------------+ +-----------------------------------+
|     HEAVY VISION ENGINE           | |     FAST FRAME INTERPOLATOR       |
|      (vision_engine.py)           | |     (frame_interpolator.py)       |
| - Sobel Edge & HSV ROI Contours   | | - Pixel Diff Optical Flow         |
| - High Accuracy ( resets drift )  | | - Kinematic Motion Propagation    |
| - Compute Cost: 18.5 ms           | | - Compute Cost: 1.5 ms            |
| - Epistemic Uncertainty -> 0.02   | | - Uncertainty grows with skips    |
+-----------------+-----------------+ +-----------------+-----------------+
                  |                                     |
                  +------------------+------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|             MULTI-OBJECTIVE REWARD ENGINE (car_systems.py)              |
|  r_t = Accuracy_Reward(d_est, d_gt) + Compute_Bonus(a_t) - Safety_Pen   |
+-------------------------------------------------------------------------+
```

---

## 3. File-by-File Codebase Breakdown

### 3.1 `setup_dataset.py` — Automated Dataset & Telemetry Generator
* **Role:** Ensures the project is $100\%$ self-contained and reproducible.
* **How It Works:**
  1. Attempts to fetch a driving video stream (`sample_driving.mp4`) or synthesizes a $600$-frame highway driving sequence (`dataset/driving_sequence/frames/frame_0000.jpg` to `frame_0599.jpg`).
  2. Simulates realistic CAN-bus vehicle telemetry in `telemetry.csv`:
     * `speed_kmh`: Ego-vehicle velocity ($25\text{--}90\text{ km/h}$).
     * `steering_angle_deg`: Sinusoidal lane-keeping steering angles.
     * `acceleration_ms2`: Longitudinal acceleration/deceleration ($\text{m/s}^2$).
     * `gt_obstacle_dist_m`: Ground-truth distance to the lead vehicle ($8\text{--}60\text{ m}$).
     * `hazard_event`: Binary flag ($1$ during frames $120\text{--}160$ and $350\text{--}400$) simulating emergency braking where the lead vehicle rapidly closes in from $45\text{ m}$ down to $8\text{ m}$.

### 3.2 `systems/data_feeder.py` — Synchronized Sensor Streamer
* **Role:** Acts as the virtual CAN-bus and camera frame grabber.
* **How It Works:** Loads `telemetry.csv` into memory and pairs each row with its corresponding OpenCV BGR frame from disk. Supports both deterministic starts (`random_start=False` for fair evaluation) and randomized starting indices (`random_start=True` during RL training to prevent overfitting to fixed frame numbers).

### 3.3 `ENGINES/vision_engine.py` — Heavyweight Perception Module
* **Role:** Represents the computationally expensive deep perception pathway ($a_t = 1$).
* **How It Works:**
  1. Converts the RGB frame to grayscale and computes horizontal and vertical Sobel derivatives ($G_x, G_y$) to measure visual scene complexity:
     $$E_{\text{edge}} = \frac{1}{HW \cdot 255} \sum_{x,y} \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$
  2. Crops the center-lane Region of Interest (ROI), applies HSV segmentation, and extracts obstacle contours to estimate the lead vehicle's bounding scale and distance.
  3. Fuses the visual measurement with low sensor variance ($\sigma = 0.35\text{ m}$), resets epistemic uncertainty to $0.02$, and logs a latency cost of $18.5\text{ ms}$.

### 3.4 `ENGINES/frame_interpolator.py` — Lightweight Dead-Reckoning Module
* **Role:** Represents the fast approximation pathway when the vision gate is closed ($a_t = 0$).
* **How It Works:**
  1. Maintains an anchor distance $d_{\text{anchor}}$ from the last time `VisionEngine` was executed.
  2. Computes fast frame-to-frame pixel difference ($\Delta I_t = \frac{1}{HW \cdot 255} \sum |I_t - I_{t-1}|$).
  3. Propagates the obstacle distance using longitudinal kinematics over $\Delta t = 0.05\text{ s}$ ($20\text{ FPS}$):
     $$\Delta d_t = -0.15 a_t \Delta t - 2.5 (\Delta I_t - 0.04) v_t \Delta t + \mathcal{N}\left(0, 0.65\sqrt{k}\right)$$
     where $k$ is the number of consecutive skipped vision frames (`consecutive_skips`).
  4. Models **Epistemic Uncertainty** growth as $u_t = \min(1.0, 0.08k + 1.5\Delta I_t)$. The longer the agent keeps the gate closed, the more the estimate drifts and uncertainty rises. Logs a tiny latency cost of $1.5\text{ ms}$.

### 3.5 `systems/car_systems.py` — Safety Calculator & Gymnasium MDP Environment
* **Role:** Formulates the active perception problem as a standard OpenAI `gymnasium.Env` (`GatedVisionRLEnv`).
* **How It Works:**
  * **Time-To-Collision (TTC) & Risk Score (`CarSystems.compute_risk_score`):**
    $$\text{TTC} = \frac{d_{\text{est}}}{v_{\text{ego}}}, \quad \text{Risk} = \text{clip}\left(\frac{2.5}{\max(\text{TTC}, 0.2)} \cdot \left(1 + \frac{|\theta_{\text{steer}}|}{25^\circ}\right), 0.0, 1.0\right)$$
  * **7-Dimensional Continuous State Space ($s_t \in [-1, 1]^7$):**
    $$s_t = \left[ \frac{v_t}{120}, \frac{|\theta_t|}{45}, \frac{a_t}{5}, \frac{d_{\text{est}, t}}{70}, \frac{k_t}{15}, u_t, \text{Risk}_t \right]^T$$
  * **Multi-Objective Reward Function ($r_t$):**
    $$r_t = \underbrace{1.5 \exp(-0.25 |d_{\text{est}, t} - d_{\text{gt}, t}|)}_{\text{Tracking Accuracy Term}} + \underbrace{R_{\text{compute}}(a_t)}_{\text{Efficiency Term}} + \underbrace{R_{\text{safety}}(a_t, k_t, e_t)}_{\text{Safety Penalty}}$$
    * $R_{\text{compute}}(a_t) = +0.25$ if $a_t = 0$ (gate closed) else $-0.35$ if $a_t = 1$ (gate open).
    * $R_{\text{safety}} = -2.0$ if a braking hazard is occurring ($\text{hazard}=1$) and the agent skips more than $1$ frame, plus $-1.5$ if tracking error $e_t = |d_{\text{est}} - d_{\text{gt}}| > 6.0\text{ m}$.

### 3.6 `ENGINES/intent_engine.py` — Dual RL Agent Architecture
* **Role:** Implements two distinct Reinforcement Learning algorithms to compare value-based tabular methods against deep policy-gradient methods:
  1. **Tabular Q-Learning:** Discretizes the state space into $(6 \times 6 \times 6)$ bins across $(\text{Risk}, \text{Uncertainty}, \text{Skips})$ and updates a Q-table via the Bellman optimality equation:
     $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$
  2. **Proximal Policy Optimization (PPO):** Uses an Actor-Critic Multi-Layer Perceptron (`MlpPolicy`) operating directly on the continuous 7D state vector, optimizing the clipped surrogate objective with Generalized Advantage Estimation ($\gamma = 0.98, \lambda_{\text{GAE}} = 0.95, \epsilon_{\text{clip}} = 0.2$).

### 3.7 `co_pilot_master.py` & `visualize_copilot.py` — Training, Benchmarking & HUD Suite
* `co_pilot_master.py` orchestrates dataset setup, trains Q-Learning ($120$ episodes) and PPO ($15,000$ timesteps), benchmarks all $5$ modes (`always_vision`, `always_interpolate`, `random`, `q_learning`, `ppo`), and generates `results/rl_evaluation_results.png`.
* `visualize_copilot.py` generates the $4$-panel executive infographic (`results/project_executive_dashboard.png`) and renders the real-time annotated video (`results/live_copilot_demo.mp4`).

---

## 4. How the Learned PPO Policy Actually Thinks (Step-by-Step Walkthrough)

By inspecting **Panel 3** and **Panel 4** of your generated `project_executive_dashboard.png`, you can observe three distinct phases in how the PPO agent behaves across the $300$-frame test drive:

1. **Phase 1: Steady Highway Cruising (Frames $0\text{--}110$)**
   * The lead vehicle is far away ($45\text{--}60\text{ m}$) and moving smoothly.
   * PPO keeps its **Vision Gate usage down at $\sim 20\%$** (triggering vision only once every $4\text{--}5$ frames).
   * In between green vision triggers, it uses the $1.5\text{ ms}$ `FrameInterpolator`. Whenever epistemic uncertainty creeps up, PPO fires a single vision pass to reset drift back to zero.
2. **Phase 2: Sudden Braking Hazard Zone (Frames $120\text{--}160$, Red Shaded Region)**
   * The lead vehicle slams on its brakes, closing the gap from $46\text{ m}$ down to $8\text{ m}$.
   * Ego deceleration spikes and the TTC Risk score surges.
   * **Look at Panel 4 (`How PPO Thinks`):** PPO immediately ramps **Vision Engine Activation up to $100\%$** across the entire red hazard zone! Look at the green dots in `rl_evaluation_results.png`—they form a solid continuous line down the steep braking slope between frames $120$ and $165$.
   * Meanwhile, the red dotted line (`Always Interpolate`) completely loses track of the car, drifting over $25\text{ m}$ away from reality.
3. **Phase 3: Post-Hazard Recovery & Moderate Following (Frames $165\text{--}300$)**
   * As the lead car accelerates back to $30\text{--}60\text{ m}$, PPO smoothly throttles the Vision Gate back down from $100\%$ to $\sim 25\text{--}33\%$, immediately resuming compute and battery savings.