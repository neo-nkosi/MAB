import gym
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from gym.wrappers import RecordVideo

# Helper functions and classes
class IHT:
    "Structure to handle collisions"

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles_list = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles_list.append(hash_coords(coords, iht_or_size, read_only))
    return tiles_list


# Boundaries for position and velocity in MountainCar environment
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


# Wrapper class for state-action value function
class ValueFunction:
    def __init__(self, alpha, n_actions, num_of_tilings=8, max_size=2048):
        self.action_space = gym.spaces.Discrete(n_actions)
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.step_size = alpha / num_of_tilings
        self.hash_table = IHT(max_size)
        self.weights = np.zeros(max_size)
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    def _get_active_tiles(self, position, velocity, action):
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                             [self.position_scale * position, self.velocity_scale * velocity],
                             [action])
        return active_tiles

    # Estimate the value of given state and action
    def __call__(self, state, action):
        # Explicitly extract position and velocity from the state
        position, velocity = float(state[0]), float(state[1])
        # Fix for floating-point comparison with np.isclose
        if np.isclose(position, POSITION_MAX):
            return 0.0
        active_tiles = self._get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # Learn with given state, action and target
    def update(self, target, state, action):
        active_tiles = self._get_active_tiles(state[0], state[1], action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return self.action_space.sample()
        return np.argmax([self(state, action) for action in range(self.action_space.n)])


def semi_gradient_sarsa(env, value_function, num_episodes, alpha, gamma, epsilon):
    steps_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = value_function.act(state, epsilon)
        step_count = 0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = value_function.act(next_state, epsilon)

            target = reward + gamma * value_function(next_state, next_action)
            value_function.update(target, state, action)

            state, action = next_state, next_action
            step_count += 1

            if done:
                steps_per_episode.append(step_count)
                break

    return steps_per_episode


# Main program
if __name__ == '__main__':
    num_episodes = 500
    num_runs = 100
    alpha = 0.1  #lr
    epsilon = 0.1  #eps-greedy policy
    gamma = 1.0   #discount
    max_size = 2048

    all_steps = []

    for run in range(num_runs):
        env = gym.make('MountainCar-v0')
        n_actions = env.action_space.n

        value_function = ValueFunction(alpha, n_actions, max_size=max_size)

        steps = semi_gradient_sarsa(env, value_function, num_episodes, alpha, gamma, epsilon)
        all_steps.append(steps)
        print(f'Run {run + 1}/{num_runs} completed.')

        env.close()

    all_steps = np.array(all_steps)  # Shape: (num_runs, num_episodes)

    avg_steps = np.mean(all_steps, axis=0)

    plt.plot(range(num_episodes), avg_steps)
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps (log scale)')
    plt.title('Average Steps per Episode over 100 Runs (log scale)')
    plt.show()

    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n

    value_function = ValueFunction(alpha, n_actions, max_size=max_size)

    steps = semi_gradient_sarsa(env, value_function, num_episodes, alpha, gamma, epsilon)

    env.close()

    render_env = gym.make('MountainCar-v0', render_mode='rgb_array')
    render_env = RecordVideo(render_env, video_folder='videos/', episode_trigger=lambda episode_id: True)

    #reset the environment before starting recording
    state, _ = render_env.reset()
    done = False
    while not done:
        action = value_function.act(state, epsilon=0)  # ε=0 for greedy policy
        state, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
    render_env.close()
