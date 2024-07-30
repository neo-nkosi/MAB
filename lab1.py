import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#Multi armed bandit
class MultiArmedBandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.random.normal(0, 3, k)
        self.q_values = np.zeros(k)
        self.arm_counts = np.zeros(k)

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)

    def update_q_value(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class MultiArmedBanditOptimistic:
    def __init__(self, k=10, initial_value=5):
        self.k = k
        # set a mean of 0, variance of 3 for the rewards each arm
        self.means = np.random.normal(0, 3, k)
        self.q_values = np.full(k, initial_value)  # optimistic initial q-values for each arm
        self.arm_counts = np.zeros(k)  # number of times each arm is pulled

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)

    def update_q_value(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]


#Epsilon Greedy
def epsilon_greedy(mab, epsilon=0.1, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        if np.random.rand() < epsilon:
            #explore
            arm = np.random.randint(mab.k)
        else:
            #exploit
            arm = np.argmax(mab.q_values)

        reward = mab.pull(arm)
        mab.update_q_value(arm, reward)
        rewards[step] = reward
    return rewards

def simulate_e_greedy(k=10, epsilon=0.1, steps=1000, runs=100):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        mab = MultiArmedBandit(k)
        rewards = epsilon_greedy(mab, epsilon, steps)
        all_rewards[run] = rewards
    return all_rewards


#Greedy with Optimistic Initialisation
def greedy_optimistic(mab, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        arm = np.argmax(mab.q_values)

        reward = mab.pull(arm)
        mab.update_q_value(arm, reward)
        rewards[step] = reward
    return rewards


def simulate_optimistic(k=10, initial_value=5, steps=1000, runs=100):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        mab = MultiArmedBanditOptimistic(k, initial_value)
        rewards = greedy_optimistic(mab, steps)
        all_rewards[run] = rewards
    return all_rewards


def ucb(mab, c=2, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        ucb_values = mab.q_values + c * np.sqrt(np.log(step + 1) / (mab.arm_counts + 1e-5))
        arm = np.argmax(ucb_values)

        reward = mab.pull(arm)
        mab.update_q_value(arm, reward)
        rewards[step] = reward
    return rewards

def simulate_ucb(k=10, steps=1000, runs=100, epsilon=0.1):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        mab = MultiArmedBandit(k)
        rewards = ucb(mab, epsilon, steps)
        all_rewards[run] = rewards
    return all_rewards


def run_algorithm(algorithm, param, k=10, steps=1000, runs=100):
    total_reward = 0
    for _ in range(runs):
        mab = MultiArmedBandit(k)
        if algorithm.__name__ == 'epsilon_greedy':
            rewards = algorithm(mab, epsilon=param, steps=steps)
        elif algorithm.__name__ == 'ucb':
            rewards = algorithm(mab, c=param, steps=steps)
        elif algorithm.__name__ == 'greedy_optimistic':
            mab.q_values = np.full(k, param)  # Set initial Q-values
            rewards = algorithm(mab, steps=steps)
        total_reward += np.mean(rewards)
    return total_reward / runs


def plot_comparison(e_greedy_rewards, optimistic_rewards, ucb_rewards, steps=1000, window=100):
    plt.figure(figsize=(12, 6))

    for rewards, label in zip([e_greedy_rewards, optimistic_rewards, ucb_rewards],
                              ['ε-Greedy (ε=0.1)', 'Greedy with Optimistic Init (Q1=5)', 'UCB (c=2)']):
        averaged_rewards = np.mean(rewards, axis=0)
        smoothed_rewards = np.convolve(averaged_rewards, np.ones(window) / window, mode='valid')
        plt.plot(smoothed_rewards, label=label)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Multi-Armed Bandit Algorithms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mab_comparison.png')
    plt.show()


# Parameters
k = 10
steps = 1000
runs = 100

# Simulate each algorithm
e_greedy_rewards = simulate_e_greedy(k=k, epsilon=0.1, steps=steps, runs=runs)
optimistic_rewards = simulate_optimistic(k=k, initial_value=5, steps=steps, runs=runs)
ucb_rewards = simulate_ucb(k=k, steps=steps, runs=runs)

# Plot comparison over time
plot_comparison(e_greedy_rewards, optimistic_rewards, ucb_rewards, steps=steps)


def plot_summary_comparison(algorithms, param_ranges):
    plt.figure(figsize=(12, 8))

    colors = ['red', 'blue', 'black']
    labels = ['ε-greedy', 'UCB', 'greedy with optimistic initialization']
    markers = ['o', 's', '^']

    for i, (algo, param_range) in enumerate(zip(algorithms, param_ranges)):
        rewards = []
        for param in param_range:
            avg_reward = run_algorithm(algo, param)
            rewards.append(avg_reward)

        # Create a smooth curve using interpolation
        x_smooth = np.logspace(np.log10(param_range[0]), np.log10(param_range[-1]), 200)
        f = interpolate.interp1d(param_range, rewards, kind='cubic')
        y_smooth = f(x_smooth)

        # Plot both the original points and the smooth curve
        plt.plot(x_smooth, y_smooth, color=colors[i], label=labels[i])
        plt.scatter(param_range, rewards, color=colors[i], marker=markers[i], s=30, alpha=0.6)

    plt.xscale('log')
    plt.xlabel('ε / c / Q₀')
    plt.ylabel('Average reward over first 1000 steps')
    plt.title('Summary comparison of algorithms')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set y-axis limits
    y_min = 0
    y_max = 5
    plt.ylim(y_min - 0.1, y_max + 0.1)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('mab_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#parameter ranges and number of points
algorithms = [epsilon_greedy, ucb, greedy_optimistic]
param_ranges = [
    np.logspace(-3, 0, 20),  # for ε-greedy
    np.logspace(-1, 2, 20),  # for UCB
    np.logspace(0, 2, 20)  # for optimistic initialization
]

plot_summary_comparison(algorithms, param_ranges)