# replay_buffer.py

import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        """
        Initialize the replay buffer.

        Parameters:
            size (int): Maximum number of transitions to store in the buffer.
        """
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.

        Parameters:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state after taking the action.
            done: Whether the episode has terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
            batch_size (int): The number of transitions to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        # Stack the states and next_states along the first dimension
        states = np.stack([np.array(transition[0], copy=False) for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.stack([np.array(transition[3], copy=False) for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
