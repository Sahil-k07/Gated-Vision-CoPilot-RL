import os
import numpy as np
from stable_baselines3 import PPO


class IntentEngineRL:
    """
    Reinforcement Learning Gating Agent (Intent Engine).
    Supports both Deep RL (PPO) and Tabular Q-Learning for academic benchmarking.
    """

    def __init__(self, env, model_dir="models"):
        self.env = env
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ppo_path = os.path.join(self.model_dir, "ppo_gated_copilot")

        self.ppo_model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0
        )

        # Tabular Q-table over discretized (risk_bin, uncertainty_bin, skip_bin) -> 2 actions
        self.q_table = np.zeros((6, 6, 6, 2), dtype=np.float32)

    def _discretize_obs(self, obs: np.ndarray) -> tuple:
        skips_bin = min(5, int(obs[4] * 6))
        uncert_bin = min(5, int(obs[5] * 6))
        risk_bin = min(5, int(obs[6] * 6))
        return (risk_bin, uncert_bin, skips_bin)

    def train_q_learning(self, episodes=120, alpha=0.15, gamma=0.98, epsilon_start=1.0):
        """Trains the explicit Tabular Q-Learning agent for RL coursework reporting."""
        epsilon = epsilon_start
        reward_history = []

        for ep in range(episodes):
            obs, _ = self.env.reset(options={"random_start": True})
            state = self._discretize_obs(obs)
            done = False
            ep_reward = 0.0

            while not done:
                if np.random.rand() < epsilon:
                    action = int(self.env.action_space.sample())
                else:
                    action = int(np.argmax(self.q_table[state]))

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretize_obs(next_obs)
                done = terminated or truncated

                # Bellman Q-Update
                best_next_q = np.max(self.q_table[next_state])
                td_target = reward + (0.0 if done else gamma * best_next_q)
                self.q_table[state][action] += alpha * (td_target - self.q_table[state][action])

                state = next_state
                ep_reward += reward

            epsilon = max(0.05, epsilon * 0.96)
            reward_history.append(ep_reward)

        np.save(os.path.join(self.model_dir, "q_table.npy"), self.q_table)
        return reward_history

    def train_ppo(self, total_timesteps=15000):
        """Trains the Deep RL PPO Policy network."""
        self.ppo_model.learn(total_timesteps=total_timesteps, progress_bar=False)
        self.ppo_model.save(self.ppo_path)

    def predict(self, obs: np.ndarray, mode="ppo") -> int:
        if mode == "ppo":
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            return int(action)
        elif mode == "q_learning":
            state = self._discretize_obs(obs)
            return int(np.argmax(self.q_table[state]))
        elif mode == "always_vision":
            return 1
        elif mode == "always_interpolate":
            return 0
        else:
            return int(self.env.action_space.sample())