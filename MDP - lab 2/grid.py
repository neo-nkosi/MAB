
rows = 7
cols = 7
# Created a grid of pairs. 1st element is the type of block (available, obstacle, goal).
# 2nd element is the optimal value for that block
grid = [[('.',0) for _ in range(cols)] for _ in range(rows)]

# goal position
grid[0][6] = ('G',0)

# initial position
grid[6][0] = ('S',0)

# adding obstacles
for i in range(0,6):
    grid[2][i] = ('X',0)

for i in range(rows):
    for j in range(cols):
        print(grid[i][j][0],end=" ")
    print()