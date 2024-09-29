from gym import spaces
import numpy as np
import torch
import torch.optim as optim

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        device,  # Add device as a parameter
    ):
        """
        Initialize the DQN agent with the given parameters.
        """
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device  # Store the device

        # Initialize policy and target networks and move them to the correct device (GPU/CPU)
        self.policy_network = DQN(observation_space, action_space).to(self.device)
        self.target_network = DQN(observation_space, action_space).to(self.device)
        self.update_target_network()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def optimise_td_loss(self):
        """
        Optimize the TD-error over a single minibatch of transitions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to torch tensors and normalize pixel values
        states = torch.tensor(states, device=self.device, dtype=torch.float32) / 255.0
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32) / 255.0
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Compute current Q-values
        q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_actions = self.policy_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]

            # Compute target Q-values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update the target network by copying weights from the policy network.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state.
        """
        # Check if state is already a tensor and handle it properly
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(self.device).unsqueeze(0) / 255.0
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0) / 255.0

        with torch.no_grad():
            q_values = self.policy_network(state)
        action = q_values.argmax().item()
        return action
