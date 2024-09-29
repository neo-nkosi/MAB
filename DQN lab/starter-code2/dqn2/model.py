# model.py

import torch
import torch.nn as nn
from gymnasium import spaces  # Updated import

class DQN(nn.Module):
    def __init__(self, observation_space, action_space):
        """
        Initialize the DQN network.
        """
        super(DQN, self).__init__()
        assert isinstance(observation_space, spaces.Box), "observation_space must be of type Box"
        assert len(observation_space.shape) == 3, "observation space must have the form channels x width x height"
        assert isinstance(action_space, spaces.Discrete), "action_space must be of type Discrete"

        c, h, w = observation_space.shape
        num_actions = action_space.n

        # Define your convolutional neural network architecture
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute the size of the feature maps after the convolutional layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = x / 255.0  # Normalize input if observations are images
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
