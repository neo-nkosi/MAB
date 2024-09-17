import gym
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from tiles3 import  IHT, tiles


class SARSAAgent:
    def __init__(self, num_actions, num_tilings=8, tile_size=8, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_size = tile_size
        self.alpha = alpha / num_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.iht = IHT(4096)
        self.w = np.zeros(4096)

    def get_tiles(self, state):
        position, velocity = state
        return tiles(self.iht, self.num_tilings,
                     [self.tile_size * position / 1.2, self.tile_size * velocity / 0.07])

    def q_value(self, state, action):
        return np.sum(self.w[self.get_tiles(state) + [action]])

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = [self.q_value(state, a) for a in range(self.num_actions)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, done):
        active_tiles = self.get_tiles(state) + [action]
        q_value = np.sum(self.w[active_tiles])
        if done:
            target = reward
        else:
            next_q_value = np.sum(self.w[self.get_tiles(next_state) + [next_action]])
            target = reward + self.gamma * next_q_value
        delta = target - q_value
        self.w[active_tiles] += self.alpha * delta

# Training function
def train_sarsa(env, agent, num_episodes):
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = agent.choose_action(state)
        episode_length = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Explicitly check if terminated and truncated are of boolean type
            if not isinstance(terminated, bool):
                terminated = bool(terminated)
            if not isinstance(truncated, bool):
                truncated = bool(truncated)

            done = terminated or truncated
            episode_length += 1

            if done:
                agent.update(state, action, reward, next_state, None, done)
                episode_lengths.append(episode_length)
                break

            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

    return episode_lengths


# Main execution
def run_experiment(num_runs=100, num_episodes=500):
    env = gym.make('MountainCar-v0')
    all_episode_lengths = []

    for run in range(num_runs):
        agent = SARSAAgent(env.action_space.n)
        episode_lengths = train_sarsa(env, agent, num_episodes)
        all_episode_lengths.append(episode_lengths)
        print(f"Run {run+1}/{num_runs} completed")

    avg_episode_lengths = np.mean(all_episode_lengths, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), avg_episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.title('SARSA Learning Curve (Average over 100 runs)')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    return agent  # Return the last trained agent for rendering

# Run the experiment
trained_agent = run_experiment()

# Render the learned policy
def render_policy(env, agent):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

# Uncomment the following line to render the learned policy
# render_policy(gym.make('MountainCar-v0', render_mode='human'), trained_agent)