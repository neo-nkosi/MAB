import numpy as np

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.reward_distributions = np.random.normal(0, 3, k)

    def pull(self, arm):
        return np.random.normal(self.reward_distributions[arm], 1)


class EpsilonGreedy:
    def __init__(self, mab, epsilon=0.1):
        self.mab = mab
        self.epsilon = epsilon
        self.q_values = np.zeros(mab.k)
        self.n_pulls = np.zeros(mab.k)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.mab.k)
        else:
            return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

class OptimisticGreedy:
    def __init__(self, mab, initial_value=5):
        self.mab = mab
        self.q_values = np.full(mab.k, initial_value)
        self.n_pulls = np.zeros(mab.k)

    def select_arm(self):
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

class UCB:
    def __init__(self, mab, c=2):
        self.mab = mab
        self.c = c
        self.q_values = np.zeros(mab.k)
        self.n_pulls = np.zeros(mab.k)
        self.t = 0

    def select_arm(self):
        self.t += 1
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / (self.n_pulls + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

# 10 arm bandit
mab = MultiArmedBandit(10)
eg = EpsilonGreedy(mab)
og = OptimisticGreedy(mab)
ucb = UCB(mab)
