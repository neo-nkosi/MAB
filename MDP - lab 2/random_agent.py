import numpy as np
import matplotlib.pyplot as plt

rows = 7
cols = 7
# Created a grid of pairs. 1st element is the type of block (available, obstacle, goal).
# 2nd element is the reward of that block
grid = [[('.', -1) for _ in range(cols)] for _ in range(rows)]

# goal position
goal_position = (0, 0)
grid[0][6] = ('G', 20)

# initial position
initial_position = (6, 0)
grid[6][0] = ('S', 0)

# adding obstacles
for i in range(0, 6):
    grid[2][i] = ('X', 0)

# Print the grid
for i in range(rows):
    for j in range(cols):
        print(grid[i][j][0], end=" ")
    print()

actions = {
    'N': (-1, 0),
    'E': (0, 1),
    'S': (1, 0),
    'W': (0, -1)
}

def execute_action(action, state):
    x, y = state
    dx, dy = action
    new_state = (x + dx, y + dy)

    # check if the action does not move the agent out of bounds or into an obstacle
    if (0 <= new_state[0] < rows and 0 <= new_state[1] < cols) and (grid[new_state[0]][new_state[1]][0] != 'X'):
        return new_state, grid[new_state[0]][new_state[1]][1]
    else:
        # agent moves out of bounds or into an obstacle, keep agent at same position with -1 penalty
        return state, -1

def random_agent(max_steps=50):
    current_state = initial_position
    total_return = 0

    for _ in range(max_steps):
        if current_state == goal_position:
            break

        action = np.random.choice(list(actions.keys()))
        new_state, reward = execute_action(actions[action], current_state)

        total_return += reward
        current_state = new_state

    return total_return

def run_simulations(num_runs=20, max_steps=50):
    returns = []
    for _ in range(num_runs):
        returns.append(random_agent(max_steps))
    return returns

def plot_returns(returns):
    plt.bar(range(len(returns)), returns)
    plt.xlabel('Run')
    plt.ylabel('Total Return')
    plt.title('Returns Accumulated by Random Agent over 20 Runs')
    plt.show()

# Running the simulations and plotting the results
returns = run_simulations()
plot_returns(returns)
