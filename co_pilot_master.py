import os
import numpy as np
import matplotlib.pyplot as plt

from setup_dataset import download_or_generate_dataset, TELEMETRY_FILE
from systems.car_systems import GatedVisionRLEnv
from ENGINES.intent_engine import IntentEngineRL


def evaluate_policy(env: GatedVisionRLEnv, agent: IntentEngineRL, mode: str) -> dict:
    obs, _ = env.reset(options={"random_start": False})
    done = False

    total_reward = 0.0
    errors = []
    compute_times = []
    actions = []
    est_traj, gt_traj = [], []

    while not done:
        action = agent.predict(obs, mode=mode)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        errors.append(info["tracking_error_m"])
        compute_times.append(info["compute_ms"])
        actions.append(info["action_gate"])
        est_traj.append(info["est_dist_m"])
        gt_traj.append(info["gt_dist_m"])

    return {
        "mode": mode,
        "total_reward": round(total_reward, 2),
        "mae_m": round(float(np.mean(errors)), 3),
        "avg_latency_ms": round(float(np.mean(compute_times)), 2),
        "vision_gate_rate_pct": round(float(np.mean(actions) * 100.0), 1),
        "est_traj": est_traj,
        "gt_traj": gt_traj,
        "actions": actions
    }


def save_evaluation_plots(q_rewards, eval_results):
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Plot 1: Q-Learning Training Convergence
    window = 10
    smoothed = np.convolve(q_rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(q_rewards, alpha=0.35, color="steelblue", label="Episode Reward")
    axes[0].plot(range(window - 1, len(q_rewards)), smoothed, color="navy", linewidth=2, label="10-Ep Moving Avg")
    axes[0].set_title("RL Gating Agent Training Convergence (Q-Learning)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative MDP Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Obstacle Distance Tracking & Gating Triggers (PPO)
    ppo_res = eval_results["ppo"]
    steps = np.arange(len(ppo_res["gt_traj"]))
    axes[1].plot(steps, ppo_res["gt_traj"], label="Ground Truth Obstacle Dist (m)", color="black", linewidth=2)
    axes[1].plot(steps, ppo_res["est_traj"], label="PPO Gated-Vision Estimate (m)", color="crimson", linestyle="--")

    gate_open_steps = [i for i, a in enumerate(ppo_res["actions"]) if a == 1]
    gate_open_vals = [ppo_res["est_traj"][i] for i in gate_open_steps]
    axes[1].scatter(gate_open_steps, gate_open_vals, color="limegreen", s=18, zorder=5, label="Vision Gate Triggered (a=1)")

    axes[1].set_title("PPO CoPilot Obstacle Tracking vs. Active Vision Gating")
    axes[1].set_xlabel("Frame Timestep")
    axes[1].set_ylabel("Distance (m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join("results", "rl_evaluation_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[Plot Saved] Evaluation charts saved to '{plot_path}'")


def main():
    print("=" * 68)
    print("   GATED-VISION-COPILOT: REINFORCEMENT LEARNING FRAMEWORK")
    print("=" * 68)

    # Step 1: Ensure Dataset is Installed
    if not os.path.exists(TELEMETRY_FILE):
        download_or_generate_dataset(num_frames=600)

    # Step 2: Initialize RL Environment & Intent Agent
    env = GatedVisionRLEnv(max_episode_steps=300)
    agent = IntentEngineRL(env)

    # Step 3: Train Tabular Q-Learning & Deep PPO Policy
    print("\n[1/3] Training Tabular Q-Learning Gating Agent (120 Episodes)...")
    q_rewards = agent.train_q_learning(episodes=120)

    print("[2/3] Training Deep RL (PPO) Gating Policy (15,000 Timesteps)...")
    agent.train_ppo(total_timesteps=15000)

    # Step 4: Evaluate & Compare Against Baselines
    print("[3/3] Evaluating Policies across Test Driving Sequence...")
    modes = ["always_vision", "always_interpolate", "random", "q_learning", "ppo"]
    results = {}
    for m in modes:
        results[m] = evaluate_policy(env, agent, mode=m)

    print("\n" + "=" * 78)
    print(f"{'Policy Mode':<20} | {'Reward':>8} | {'MAE (m)':>8} | {'Latency (ms)':>12} | {'Vision Gate %':>13}")
    print("-" * 78)
    for m in modes:
        r = results[m]
        print(
            f"{r['mode']:<20} | {r['total_reward']:>8.2f} | {r['mae_m']:>8.3f} | "
            f"{r['avg_latency_ms']:>12.2f} | {r['vision_gate_rate_pct']:>12.1f}%"
        )
    print("=" * 78)

    save_evaluation_plots(q_rewards, results)


if __name__ == "__main__":
    main()