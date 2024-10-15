import numpy as np
import matplotlib.pyplot as plt

# Group member names and student numbers
# Neo Nkosi - 2437872
# Joshua Moorhead - 2489197
# Naomi Muzamani - 2456718
# PraiseGod Emenike - 2428608

# Gridworld setup
rows, cols = 7, 7
grid = [[('.', -1) for _ in range(cols)] for _ in range(rows)]
goal_position = (0, 0)
initial_position = (6, 0)
grid[0][6] = ('G', 20)
grid[6][0] = ('S', 0)
for i in range(0, 6):
    grid[2][i] = ('X', 0)

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
    if (0 <= new_state[0] < rows and 0 <= new_state[1] < cols) and (grid[new_state[0]][new_state[1]][0] != 'X'):
        return new_state, grid[new_state[0]][new_state[1]][1]
    else:
        return state, -1

# Random agent
def random_agent(max_steps=50):
    current_state = initial_position
    total_return = 0
    trajectory = [current_state]

    for _ in range(max_steps):
        if current_state == goal_position:
            break
        action = np.random.choice(list(actions.keys()))
        new_state, reward = execute_action(actions[action], current_state)
        total_return += reward
        current_state = new_state
        trajectory.append(current_state)

    return total_return, trajectory

# Greedy agent using optimal value function
v_star = [
    [20, 19, 18, 17, 16, 15, 14],
    [19, 18, 17, 16, 15, 14, 13],
    [0, 0, 0, 0, 0, 0, 12],
    [5, 6, 7, 8, 9, 10, 11],
    [4, 5, 6, 7, 8, 9, 10],
    [3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8]
]

def greedy_policy(state):
    best_action = None
    best_value = float('-inf')

    for action in actions.values():
        next_state, _ = execute_action(action, state)
        value = v_star[next_state[0]][next_state[1]]
        if value > best_value:
            best_value = value
            best_action = action

    return best_action

def greedy_agent(max_steps=50):
    state = initial_position
    total_reward = 0
    trajectory = [state]

    for _ in range(max_steps):
        if state == goal_position:
            break
        action = greedy_policy(state)
        new_state, reward = execute_action(action, state)
        total_reward += reward
        state = new_state
        trajectory.append(state)

    return total_reward, trajectory

# Running simulations
def run_simulations(agent_fn, num_runs=20, max_steps=50):
    returns = []
    trajectories = []
    for _ in range(num_runs):
        total_return, trajectory = agent_fn(max_steps)
        returns.append(total_return)
        trajectories.append(trajectory)
    return returns, trajectories

# Plotting
def plot_returns(random_returns, greedy_returns):
    plt.figure(figsize=(10, 5))
    plt.bar(['Random Agent', 'Greedy Agent'], [np.mean(random_returns), np.mean(greedy_returns)])
    plt.xlabel('Agent Type')
    plt.ylabel('Average Return')
    plt.title('Average Returns of Agents over 20 Runs')
    plt.show()

def plot_trajectories(random_trajectory, greedy_trajectory):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    def plot_grid(ax, grid, trajectory, title):
        for i in range(rows):
            for j in range(cols):
                color = 'white' if grid[i][j][0] != 'X' else 'gray'
                ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, fill=True, facecolor=color, edgecolor='black'))
                ax.text(j + 0.5, rows - 1 - i + 0.5, f"{grid[i][j][1]}", ha='center', va='center')

        path_x = [state[1] + 0.5 for state in trajectory]
        path_y = [rows - 1 - state[0] + 0.5 for state in trajectory]
        ax.plot(path_x, path_y, 'r-', linewidth=2, markersize=8, marker='o')

        ax.plot(path_x[0], path_y[0], 'ro', markersize=12, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'o', markersize=12, label='End')
        ax.plot(6 + 0.5, rows - 1 - 0 + 0.5, 'go', markersize=12, label='Goal')

        ax.set_title(title)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols))
        ax.set_yticklabels(range(rows - 1, -1, -1))
        ax.grid(True)
        ax.legend()

    plot_grid(axs[0], grid, random_trajectory, "Random Agent Trajectory")
    plot_grid(axs[1], grid, greedy_trajectory, "Greedy Agent Trajectory")
    plt.show()

# Main
random_returns, random_trajectories = run_simulations(random_agent)
greedy_returns, greedy_trajectories = run_simulations(greedy_agent)

plot_returns(random_returns, greedy_returns)
plot_trajectories(random_trajectories[0], greedy_trajectories[0])
