import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# create environment
env = gym.make('CliffWalking-v0')

epsilon = 0.1
alpha = 0.5
gamma = 1.0
num_episodes = 200
num_runs = 100


def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])


def sarsa_lambda(env, num_episodes, lambda_value):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    e = np.zeros((env.observation_space.n, env.action_space.n))
    episode_returns = np.zeros(num_episodes)

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            e[state, action] += 1

            Q += alpha * delta * e
            e *= gamma * lambda_value

            state = next_state
            action = next_action
            total_reward += reward

        episode_returns[episode] = total_reward

    return Q, episode_returns


lambda_values = [0, 0.3, 0.5]

# initialization
all_returns = {}
for lambda_val in lambda_values:
    all_returns[lambda_val] = np.zeros((num_runs, num_episodes))

all_Q_values = {}
for lambda_val in lambda_values:
    all_Q_values[lambda_val] = []

for lambda_val in lambda_values:
    for run in tqdm(range(num_runs), desc=f"Lambda {lambda_val}"):
        Q, returns = sarsa_lambda(env, num_episodes, lambda_val)
        all_returns[lambda_val][run] = returns
        all_Q_values[lambda_val].append(Q)

avg_returns = {}
std_returns = {}
for lambda_val in lambda_values:
    avg_returns[lambda_val] = np.mean(all_returns[lambda_val], axis=0)
    std_returns[lambda_val] = np.std(all_returns[lambda_val], axis=0)

plt.figure(figsize=(10, 6))
for lambda_val in lambda_values:
    plt.plot(avg_returns[lambda_val], label=f'λ = {lambda_val}')

plt.ylim(bottom=-1000)

plt.xlabel('Episode')
plt.ylabel('Average Return')
plt.title('SARSA(λ) Performance on CliffWalking')
plt.legend()
plt.savefig('sarsa_lambda_performance.png')
plt.close()


# animation of value function heatmaps (GIF)
def create_heatmap_animation(Q_values):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Value Function Heatmaps')

    heatmaps = []
    for i, lambda_val in enumerate(lambda_values):
        state_values = np.max(Q_values[lambda_val][0], axis=1).reshape(4, 12)
        heatmap = axs[i].imshow(state_values, cmap='hot', animated=True)
        axs[i].set_title(f'λ = {lambda_val}')
        heatmaps.append(heatmap)

    def update(frame):
        for i, lambda_val in enumerate(lambda_values):
            state_values = np.max(Q_values[lambda_val][frame], axis=1).reshape(4, 12)
            heatmaps[i].set_array(state_values)
        return heatmaps

    anim = FuncAnimation(fig, update, frames=num_runs, interval=200, blit=True)
    anim.save('value_function_heatmaps.gif', writer='pillow', fps=5)


create_heatmap_animation(all_Q_values)

print("Execution completed. Check the output files:")
print("1. sarsa_lambda_performance.png - Combined plot of average returns")
print("2. value_function_heatmaps.gif - Animation of value function heatmaps")