import matplotlib.pyplot as plt

rows, cols = 7, 7

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

# Set V* values
v_star = [
    [20, 19, 18, 17, 16, 15, 14],
    [19, 18, 17, 16, 15, 14, 13],
    [0, 0, 0, 0, 0, 0, 12],
    [5, 6, 7, 8, 9, 10, 11],
    [4, 5, 6, 7, 8, 9, 10],
    [3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8]
]

for i in range(rows):
    for j in range(cols):
        if grid[i][j][0] != 'X':
            grid[i][j] = (grid[i][j][0], v_star[i][j])

# Print the grid
for i in range(rows):
    for j in range(cols):
        print(f"{grid[i][j][0]},{grid[i][j][1]:<2}", end=" ")
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


def greedy_policy(state):
    best_action = None
    best_value = float('-inf')

    for action in actions.values():
        next_state, _ = execute_action(action, state)
        value = grid[next_state[0]][next_state[1]][1]
        if value > best_value:
            best_value = value
            best_action = action

    return best_action


def run_episode(start_state=initial_position, max_steps=50):
    state = start_state
    path = [state]
    total_reward = 0

    for _ in range(max_steps):
        action = greedy_policy(state)
        new_state, reward = execute_action(action, state)

        path.append(new_state)
        total_reward += reward
        state = new_state

        if state == goal_position:
            break

    return path, total_reward


# Run the greedy agent
greedy_path, total_reward = run_episode()


# Plot the grid world and agent's path
def plot_grid_world_with_path(grid, path):
    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(rows):
        for j in range(cols):
            color = 'white' if grid[i][j][0] != 'X' else 'gray'
            ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, fill=True, facecolor=color, edgecolor='black'))
            ax.text(j + 0.5, rows - 1 - i + 0.5, f"{grid[i][j][1]}", ha='center', va='center')

    path_x = [state[1] + 0.5 for state in path]
    path_y = [rows - 1 - state[0] + 0.5 for state in path]
    ax.plot(path_x, path_y, 'r-', linewidth=2, markersize=8, marker='o')

    ax.plot(path_x[0], path_y[0], 'ro', markersize=12, label='Start')
    ax.plot(path_x[-1], path_y[-1], 'go', markersize=12, label='Goal')

    ax.set_title("Grid World with Agent Using Optimal Value Function")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows - 1, -1, -1))
    ax.grid(True)
    ax.legend()

    plt.show()

# Plot the grid world with the agent's path
plot_grid_world_with_path(grid, greedy_path)

# print("\nGreedy agent's path:")
# for step, state in enumerate(greedy_path):
#     print(f"Step {step}: {state}")
#
# print(f"\nTotal reward: {total_reward}")

# Example of manual input (uncomment to use)
# current_state = initial_position
# action = input("Please enter move (N/E/S/W): ")
# new_state, reward = execute_action(actions[action], current_state)
# print("New state:", new_state)
# print("Reward:", reward)