import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import make_atari, wrap_deepmind

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.device_count())  # To check how many GPUs are available
print(torch.cuda.get_device_name(0))  # To check the GPU name

env_name = 'PongNoFrameskip-v4'

# Create the Atari environment with the appropriate wrappers
env = make_atari(env_name)
env = wrap_deepmind(env, frame_stack=True, scale=False)  # Set scale=False since we're normalizing in the agent

# Initialize the replay buffer
replay_buffer = ReplayBuffer(size=100000)

# Initialize the DQN agent, ensure it runs on the correct device
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    replay_buffer=replay_buffer,
    use_double_dqn=True,
    lr=1e-4,
    batch_size=32,
    gamma=0.99,
    device=device  # Pass the device to the agent
)

# Training parameters
num_episodes = 100
target_update_interval = 10  # Update target network every 10 episodes
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
rewards = []
losses = []

for episode in range(1, num_episodes + 1):
    state, info = env.reset()
    total_reward = 0
    t = 0

    # Move state to device (GPU/CPU)
    state = torch.tensor(state, device=device, dtype=torch.float32)

    while True:
        t += 1

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Move next_state to device
        next_state = torch.tensor(next_state, device=device, dtype=torch.float32)

        # Add experience to replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, float(done))

        # Optimize the model
        loss = agent.optimise_td_loss()
        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

    # Update the target network at specified intervals
    if episode % target_update_interval == 0:
        agent.update_target_network()

    print(f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f}")

# Plotting the reward curve
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title("Training Reward Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

# Plotting the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()

# Save the trained model to the GPU-compatible format
torch.save(agent.policy_network.state_dict(), "dqn_pong_model.pth")
