import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from lightsim2grid import LightSimBackend
import numpy as np
import matplotlib.pyplot as plt


class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"
        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward

        p = Parameters()
        p.MAX_SUB_CHANGED = 4
        p.MAX_LINE_STATUS_CHANGED = 4

        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        cr.initialize(self._g2op_env)

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space)

        self.setup_observations()
        self.setup_actions()

    def setup_observations(self):
        print("Setting up observation space.")
        self.observation_space = self._gym_env.observation_space

    def setup_actions(self):
        self.action_space = self._gym_env.action_space

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()


def evaluate_agent(env, model, n_episodes=10):
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths


def plot_results(episode_rewards, episode_lengths):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    axs[0].plot(episode_rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(episode_lengths)
    axs[1].set_title('Episode Lengths')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    plt.tight_layout()
    plt.savefig('baseline_evaluation_results.png')
    plt.close()


def main():
    env = Gym2OpEnv()
    env = Monitor(env)

    model = DQN('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=50_000)

    episode_rewards, episode_lengths = evaluate_agent(env, model)
    plot_results(episode_rewards, episode_lengths)

    print(f"Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")

    model.save("baseline_dqn_grid2op_model")


if __name__ == "__main__":
    main()