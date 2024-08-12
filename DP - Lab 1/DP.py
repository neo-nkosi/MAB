import numpy as np
import matplotlib.pyplot as plt

rows = 4
cols = 4

# Created a grid of pairs. 1st element is the type of block (available, goal).
# 2nd element is the reward of that block
grid = [[('.', -1) for _ in range(cols)] for _ in range(rows)]

# goal position
goal_position = (0, 0)
grid[0][0] = ('G', 0)  # Goal state with reward 0

# initial position
initial_position = (3, 3)  # Bottom right corner
grid[3][3] = ('S', 0)  # Marking the initial position 

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

def in_place_eval(gamma = 1.0, threshold=0.01):
    V = np.zeros((rows,cols))
    delta = float('inf')

    iterations =0
    while delta >= threshold:
        iterations+=1
        delta = 0
        for i in range(rows):
            for j in range(cols):
                if (i,j) == goal_position:
                    continue
            
                v = V[i][j]
                new_V = 0

                for action in actions.values():
                    next_state, reward, _ = execute_action(action,(i,j))
                    nx, ny = next_state
                    new_V += 0.25 * (reward + gamma * V[nx][ny])
                
                V[i][j] = new_V
                delta = max(delta, abs(v-V[i][j]))

    return V, iterations

V, iterations = in_place_eval()

print("Number of iterations until convergence:", iterations)

# Create a heatmap from the value function
plt.figure(figsize=(6,6))
plt.imshow(V, cmap='viridis', origin='upper')
plt.colorbar(label='Value')
plt.title('Heatmap of In-Place Value Evaluation')
plt.xlabel('Columns')
plt.ylabel('Rows')

for i in range(rows):
    for j in range(cols):
        plt.text(j, i, f'{V[i,j]:.2f}', ha='center', va='center', color='white')


plt.show()

