import gymnasium as gym
import numpy as np
import cv2

class Atari(gym.Wrapper):
    def __init__(self, name, action_repeat=4, size=(64, 64), grayscale=True, noops=30, life_done=False, sticky_actions=0.25):
        env = gym.make(name, frameskip=1)
        super().__init__(env)
        self.env = gym.wrappers.AtariPreprocessing(
            self.env, 
            noop_max=noops, 
            frame_skip=action_repeat, 
            screen_size=size[0], 
            terminal_on_life_loss=life_done, 
            grayscale_obs=grayscale
        )
        self.observation_space = self.env.observation_space
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(1, size[0], size[1]), dtype=np.uint8
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.observation_space.shape[0] == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.observation_space.shape[0] == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs, reward, terminated, truncated, info
