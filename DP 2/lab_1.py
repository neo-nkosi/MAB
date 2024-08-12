import numpy as np

from environments.gridworld import GridworldEnv
env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)

# get actions
actions = {
    0: 'U',
    1: 'R',
    2: 'D',
    3: 'L'
}

# trajectory using a uniform random policy
state = env.reset()  # begin at the initial state
set_of_trajectory = []

done = False
while not done:
    action = np.random.choice([0, 1, 2, 3])  # randomly choose an action
    next_state, reward, done, _ = env.step(action)
    set_of_trajectory.append((state, actions[action]))
    state = next_state

# 5x5 grid to display the trajectory
grid = [['o' for _ in range(5)] for _ in range(5)]
for (state, action) in set_of_trajectory:
    row, col = divmod(state, 5)
    grid[row][col] = action

for row in grid:
    print(' '.join(row))

# visualize the final state of the environment
env.render()
