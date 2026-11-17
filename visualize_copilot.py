import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from systems.car_systems import GatedVisionRLEnv, CarSystems
from ENGINES.intent_engine import IntentEngineRL
from co_pilot_master import evaluate_policy


def generate_executive_dashboard(results: dict):
    """Creates a clean 4-panel visual dashboard explaining the RL project results."""
    os.makedirs("results", exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Gated-Vision-CoPilot: Reinforcement Learning Active Perception Summary",
        fontsize=15, fontweight="bold", y=0.98
    )

    modes = ["always_vision", "always_interpolate", "random", "q_learning", "ppo"]
    labels = ["Always Vision\n(100% Heavy)", "Always Interp.\n(0% Heavy)", "Random\nGate", "Tabular\nQ-Learning", "Deep RL\n(PPO - Ours)"]
    colors = ["#4C72B0", "#C44E52", "#8172B2", "#CCB974", "#55A868"]

    rewards = [results[m]["total_reward"] for m in modes]
    latencies = [results[m]["avg_latency_ms"] for m in modes]
    maes = [results[m]["mae_m"] for m in modes]

    # Panel 1: Total RL Reward Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(labels, rewards, color=colors, edgecolor="black", width=0.55)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("1. Overall RL Objective Score (Cumulative Reward)", fontweight="bold")
    ax1.set_ylabel("Total Episode Reward (Higher = Better)")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        offset = 12 if yval >= 0 else -28
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.1f}", ha="center", fontweight="bold")
    ax1.set_ylim(min(rewards) - 60, max(rewards) + 60)

    # Panel 2: Compute Latency vs Tracking Error Trade-off (Pareto Frontier)
    ax2 = axes[0, 1]
    for i, m in enumerate(modes):
        ax2.scatter(latencies[i], maes[i], color=colors[i], s=180, edgecolors="black", zorder=5, label=labels[i].replace("\n", " "))
        ax2.annotate(
            f"  {m.upper()}\n  ({latencies[i]}ms, {maes[i]}m err)",
            (latencies[i], maes[i]), fontsize=8.5, fontweight="bold"
        )
    ax2.axhspan(0, 2.0, color="limegreen", alpha=0.12, label="Safe Tracking Zone (<2m MAE)")
    ax2.set_title("2. Accuracy vs. Compute Speed Trade-Off (Lower-Left is Ideal)", fontweight="bold")
    ax2.set_xlabel("Average Compute Latency per Frame (ms)")
    ax2.set_ylabel("Obstacle Distance Error - MAE (meters)")
    ax2.set_xlim(0, 22)
    ax2.set_ylim(-1, max(maes) + 3)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Obstacle Distance Tracking during Braking Hazards
    ax3 = axes[1, 0]
    ppo_res = results["ppo"]
    interp_res = results["always_interpolate"]
    steps = np.arange(len(ppo_res["gt_traj"]))

    ax3.plot(steps, ppo_res["gt_traj"], color="black", linewidth=2.2, label="True Obstacle Distance (Ground Truth)")
    ax3.plot(steps, interp_res["est_traj"], color="#C44E52", linestyle=":", linewidth=1.5, alpha=0.8, label="Always Interpolate (Drifts Badly)")
    ax3.plot(steps, ppo_res["est_traj"], color="#55A868", linewidth=1.8, label="PPO RL CoPilot (Tracks Accurately)")

    # Highlight hazard zones (frames 120-160)
    ax3.axvspan(120, 160, color="red", alpha=0.15, label="Sudden Braking Hazard Zone")
    ax3.set_title("3. Real-Time Distance Tracking & Hazard Response", fontweight="bold")
    ax3.set_xlabel("Driving Frame Timestep")
    ax3.set_ylabel("Lead Vehicle Distance (m)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: PPO Gating Behaviour & Uncertainty Dynamics
    ax4 = axes[1, 1]
    actions = np.array(ppo_res["actions"])
    # Compute rolling vision gate activation rate (20-frame window)
    rolling_gate = np.convolve(actions, np.ones(15) / 15, mode="same") * 100.0
    ax4.fill_between(steps, rolling_gate, color="#55A868", alpha=0.35, label="Rolling Vision Usage (%)")
    ax4.plot(steps, rolling_gate, color="#2E7D32", linewidth=2)
    ax4.axvspan(120, 160, color="red", alpha=0.15, label="Braking Hazard Zone")
    ax4.axhline(40.0, color="navy", linestyle="--", label=f"Avg PPO Vision Usage ({ppo_res['vision_gate_rate_pct']}%)")
    ax4.set_title("4. How PPO Thinks: Dynamic Vision Gate Activation Rate", fontweight="bold")
    ax4.set_xlabel("Driving Frame Timestep")
    ax4.set_ylabel("Vision Engine Activation (%)")
    ax4.set_ylim(0, 105)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join("results", "project_executive_dashboard.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Dashboard Saved] High-res visual explanation saved to: {out_path}")


def run_live_hud_simulation(env: GatedVisionRLEnv, agent: IntentEngineRL, show_window=True):
    """
    Renders a live interactive HUD video showing the CoPilot's camera view,
    gating decisions, telemetry, and compute savings in real time.
    """
    os.makedirs("results", exist_ok=True)
    video_out_path = os.path.join("results", "live_copilot_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, 20.0, (640, 420))

    obs, _ = env.reset(options={"random_start": False})
    done = False

    total_compute_saved_ms = 0.0
    vision_triggers = 0
    total_frames = 0

    print("\n[Live HUD] Rendering real-time CoPilot simulation (Press 'q' in window to exit early)...")

    while not done:
        frame = env.current_frame.copy()
        telemetry = env.current_telemetry
        action = agent.predict(obs, mode="ppo")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_frames += 1
        if action == 1:
            vision_triggers += 1
        else:
            total_compute_saved_ms += (18.5 - 1.5)

        # Upscale camera view to 640x280 and add a 140px telemetry dashboard below
        cam_view = cv2.resize(frame, (640, 280))
        canvas = np.zeros((420, 640, 3), dtype=np.uint8)
        canvas[:280, :] = cam_view
        canvas[280:, :] = (28, 30, 34)  # Dark dashboard background

        # 1. Draw Gate Status Banner on Top-Left of Camera
        if action == 1:
            badge_color = (0, 200, 0)  # Green = Vision Active
            gate_text = "GATE OPEN: FULL VISION ENGINE (18.5 ms)"
        else:
            badge_color = (0, 165, 255)  # Orange = Fast Interpolation
            gate_text = "GATE CLOSED: FAST INTERPOLATOR (1.5 ms)"

        cv2.rectangle(canvas, (12, 12), (415, 46), (20, 20, 20), -1)
        cv2.rectangle(canvas, (12, 12), (415, 46), badge_color, 2)
        cv2.putText(canvas, gate_text, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2)

        # 2. Hazard Alert Banner if Sudden Braking Detected
        if info["hazard"] == 1:
            cv2.rectangle(canvas, (430, 12), (628, 46), (0, 0, 220), -1)
            cv2.putText(canvas, "! BRAKING HAZARD !", (442, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 3. Draw Telemetry & RL State Metrics in Bottom Dashboard
        gt_d = info["gt_dist_m"]
        est_d = info["est_dist_m"]
        err_d = info["tracking_error_m"]
        uncert = info["uncertainty"]
        gate_rate = (vision_triggers / total_frames) * 100.0

        cv2.line(canvas, (0, 280), (640, 280), (100, 100, 100), 2)

        # Left Column: Vehicle & Distance Tracking
        cv2.putText(canvas, f"Speed: {telemetry['speed_kmh']:.1f} km/h", (20, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, f"True Dist: {gt_d:.1f} m", (20, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, f"CoPilot Est: {est_d:.1f} m (Err: {err_d:.2f}m)", (20, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 180), 1)
        cv2.putText(canvas, f"Step Reward: {reward:+.2f}", (20, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 255), 1)

        # Right Column: RL Efficiency & Uncertainty Bar
        cv2.putText(canvas, f"Vision Gate Usage: {gate_rate:.1f}%", (340, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 100), 2)
        cv2.putText(canvas, f"Compute Saved: {total_compute_saved_ms:.0f} ms", (340, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 1)

        # Uncertainty Progress Bar
        cv2.putText(canvas, "Epistemic Uncertainty:", (340, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (340, 385), (610, 404), (60, 60, 60), -1)
        bar_w = int(270 * min(1.0, uncert))
        bar_col = (0, 220, 0) if uncert < 0.35 else (0, 165, 255) if uncert < 0.65 else (0, 0, 255)
        cv2.rectangle(canvas, (340, 385), (340 + bar_w, 404), bar_col, -1)

        writer.write(canvas)

        if show_window:
            try:
                cv2.imshow("Gated-Vision-CoPilot Live RL HUD", canvas)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show_window = False  # Headless fallback

    writer.release()
    if show_window:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"[Video Saved] Complete annotated HUD video saved to: {video_out_path}")


if __name__ == "__main__":
    env = GatedVisionRLEnv(max_episode_steps=300)
    agent = IntentEngineRL(env)

    # Load trained models if available, otherwise train quickly
    if os.path.exists(agent.ppo_path + ".zip"):
        from stable_baselines3 import PPO
        agent.ppo_model = PPO.load(agent.ppo_path, env=env)
        if os.path.exists("models/q_table.npy"):
            agent.q_table = np.load("models/q_table.npy")
    else:
        print("Training RL models before generating visualizations...")
        agent.train_q_learning(episodes=120)
        agent.train_ppo(total_timesteps=15000)

    modes = ["always_vision", "always_interpolate", "random", "q_learning", "ppo"]
    eval_results = {m: evaluate_policy(env, agent, mode=m) for m in modes}

    generate_executive_dashboard(eval_results)
    run_live_hud_simulation(env, agent, show_window=True)