import numpy as np

class MultiArmedBandit:
    def __init__(self, k=10):
        self.k = k
        #set a mean of 0, variance of 3 for the rewards each arm
        self.means = np.random.normal(0, 3, k)

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)


mab = MultiArmedBandit()
#pulling first arm
reward = mab.pull(0)
print(reward)