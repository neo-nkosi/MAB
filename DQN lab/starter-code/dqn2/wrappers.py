# wrappers.py

"""
Updated wrappers compatible with Gymnasium and NumPy's Generator class.
"""

import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2

cv2.ocl.setUseOpenCL(False)


def make_atari(env_id):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id, "Environment ID must contain 'NoFrameskip'"
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(
    env,
    episode_life=True,
    clip_rewards=True,
    frame_stack=False,
    scale=False,
    pytorch_img=True,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    # Apply FrameStack before PyTorchFrame
    if frame_stack:
        env = FrameStack(env, 4)
    if pytorch_img:
        env = PyTorchFrame(env)
    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset."""
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            # Replace randint with integers
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        if "FIRE" in action_meanings:
            self.fire_action = action_meanings.index("FIRE")
        else:
            raise ValueError("Environment does not have a FIRE action.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(self.fire_action)
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """End episode when a life is lost, but only reset on true game over."""
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done
        # Check for life loss
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, _ = self.env.step(0)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame."""
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.maximum.reduce(self._obs_buffer)
        self._obs_buffer.clear()
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Clip the reward to {-1, 0, 1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84."""
        super().__init__(env)
        self.width = 84
        self.height = 84
        num_channels = 1
        new_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, num_channels), dtype=np.uint8
        )
        self.observation_space = new_space

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=-1)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low / 255.0
        high = env.observation_space.high / 255.0
        self.observation_space = spaces.Box(
            low=low.min(), high=high.max(), shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class PyTorchFrame(gym.ObservationWrapper):
    """Transpose image to channel-first format (C x H x W) for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        shp = self.observation_space.shape
        new_shape = (shp[2], shp[0], shp[1])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))
