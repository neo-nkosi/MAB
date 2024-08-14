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

# Printing the grid to ensure its initialised correctly
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
            return new_state, grid[new_state[0]][new_state[1]][1], True  # reached the goal, end the episode
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

print("Number of iterations until convergence (In-Place):", iterations)

# Heatmap for In-Place
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

def two_array_eval(gamma=1.0, threshold=0.01):
    V = np.zeros((rows, cols))
    V_new = np.zeros((rows, cols))
    delta = float('inf')
    iterations = 0
    while delta >= threshold:
        iterations += 1
        for i in range(rows):
            for j in range(cols):
                if (i, j) == goal_position:
                    V_new[i][j] = 0
                    continue
                new_V = 0
                for action in actions.values():
                    next_state, reward, _ = execute_action(action, (i, j))
                    nx, ny = next_state
                    new_V += 0.25 * (reward + gamma * V[nx][ny])
                V_new[i][j] = new_V
        delta = np.max(np.abs(V - V_new))
        V = np.copy(V_new)
    return V, iterations

# Run two-array evaluation
V_two_array, iterations_two_array = two_array_eval()
print("Number of iterations until convergence (Two-Array):", iterations_two_array)

# Heatmap for Two-Array
plt.figure(figsize=(6,6))
plt.imshow(V_two_array, cmap='viridis', origin='upper')
plt.colorbar(label='Value')
plt.title('Heatmap of Two-Array Value Evaluation')
plt.xlabel('Columns')
plt.ylabel('Rows')
for i in range(rows):
    for j in range(cols):
        plt.text(j, i, f'{V_two_array[i,j]:.2f}', ha='center', va='center', color='white')
plt.show()


# below code is to plot policy evaluation for different discount rates
discount_rates = np.logspace(-0.2, 0, num=20)

in_place_iterations = []
two_array_iterations = []

for gamma in discount_rates:
    _, in_place_iter = in_place_eval(gamma=gamma)
    _, two_array_iter = two_array_eval(gamma=gamma)
    
    in_place_iterations.append(in_place_iter)
    two_array_iterations.append(two_array_iter)

plt.figure(figsize=(10,6))
plt.plot(discount_rates, in_place_iterations, label='In-Place Evaluation')
plt.plot(discount_rates, two_array_iterations, label='Two-Array Evaluation')
plt.xscale('log')
plt.xlabel('Discount Rate')
plt.ylabel('Number of Iterations to Convergence')
plt.title('Iterations to Convergence vs Discount Rate')
plt.legend()
plt.grid(True)
plt.show()

