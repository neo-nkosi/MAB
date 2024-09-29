# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces  # Updated import
from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, observation_space, action_space, replay_buffer, use_double_dqn, lr, batch_size, gamma):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.action_space = action_space

        # Initialize policy and target networks
        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def act(self, state):
        """
        Select an action given the current state.
        """
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
        action = q_values.max(1)[1].item()
        return action

    def optimise_td_loss(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors and move to GPU
        states = torch.from_numpy(states).float().to(device)  # Shape: (batch_size, c, h, w)
        actions = torch.from_numpy(actions).long().to(device)  # Shape: (batch_size)
        rewards = torch.from_numpy(rewards).float().to(device)  # Shape: (batch_size)
        next_states = torch.from_numpy(next_states).float().to(device)  # Shape: (batch_size, c, h, w)
        dones = torch.from_numpy(dones).float().to(device)  # Shape: (batch_size)

        # Compute current Q values
        q_values = self.policy_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_network(next_states).max(1)[1]
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update the target network parameters with the policy network parameters.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())
