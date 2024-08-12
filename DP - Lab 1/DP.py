rows = 4
cols = 4

# Created a grid of pairs. 1st element is the type of block (available, goal).
# 2nd element is the reward of that block
grid = [[('.', -1) for _ in range(cols)] for _ in range(rows)]

# goal position
goal_position = (0, 0)
grid[0][0] = ('G', 0)  # Goal state with reward 0

# initial position (you can change this to whatever starting point you want)
initial_position = (3, 3)  # Bottom right corner
grid[3][3] = ('S', 0)  # Marking the initial position (optional)

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

    # check if the action does not move the agent out of bounds
    if (0 <= new_state[0] < rows and 0 <= new_state[1] < cols):
        if new_state == goal_position:
            return new_state, grid[new_state[0]][new_state[1]][1], True  # Reached the goal, end the episode
        else:
            return new_state, grid[new_state[0]][new_state[1]][1], False
    else:
        # agent moves out of bounds, keep agent at same position with -1 penalty
        return state, -1, False


current_state = initial_position
done = False

while not done:
    action = input("Please enter move (N/E/S/W): ").strip().upper()
    if action not in actions:
        print("Invalid action. Please enter N, E, S, or W.")
        continue

    new_state, reward, done = execute_action(actions[action], current_state)
    print("new state:", new_state)
    print("reward:", reward)

    current_state = new_state

    if done:
        print("Goal reached! Episode ended.")
