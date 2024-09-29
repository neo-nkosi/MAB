from gymnasium import spaces
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialize the DQN network.
        """
        super(DQN, self).__init__()
        assert isinstance(observation_space, spaces.Box), "observation_space must be of type Box"
        assert len(observation_space.shape) == 3, "observation space must have the form channels x width x height"
        assert isinstance(action_space, spaces.Discrete), "action_space must be of type Discrete"

        # Network architecture
        self.conv1 = nn.Conv2d(
            in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )

        # Calculate the size of the linear layer input
        convw = self._conv_output_size(observation_space.shape[1], [8, 4, 3], [4, 2, 1])
        convh = self._conv_output_size(observation_space.shape[2], [8, 4, 3], [4, 2, 1])
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(in_features=linear_input_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=action_space.n)

    def _conv_output_size(self, size, kernel_sizes, strides):
        for k, s in zip(kernel_sizes, strides):
            size = (size - k) // s + 1
        return size

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
