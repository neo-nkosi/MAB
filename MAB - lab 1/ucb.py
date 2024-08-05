import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, k=10):
        self.k = k
        #set a mean of 0, variance of 3 for the rewards each arm
        self.means = np.random.normal(0, 3, k)
        self.q_values = np.zeros(k)
        self.arm_counts = np.zeros(k)

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)

    def update_q_value(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]


def ucb(mab, c=2, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        ucb_values = mab.q_values + c * np.sqrt(np.log(step + 1) / (mab.arm_counts + 1e-5))
        arm = np.argmax(ucb_values)

        reward = mab.pull(arm)
        mab.update_q_value(arm, reward)
        rewards[step] = reward
    return rewards

def simulate_bandit(k=10, steps=1000, runs=100, epsilon=0.1):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        mab = MultiArmedBandit(k)
        rewards = ucb(mab, epsilon, steps)
        all_rewards[run] = rewards
    return all_rewards

def plot_results(all_rewards, steps=1000, window=100):
    averaged_rewards = np.mean(all_rewards, axis=0)
    smoothed_rewards = np.convolve(averaged_rewards, np.ones(window)/window, mode='valid')

    plt.plot(smoothed_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('UCB Algorithm Performance')
    plt.show()

# Parameters
k = 10
steps = 1000
runs = 100
epsilon = 0.1

# Simulation for UCB
c = 2
all_rewards_ucb = simulate_bandit( k, steps, runs, epsilon)
plot_results(all_rewards_ucb, steps)