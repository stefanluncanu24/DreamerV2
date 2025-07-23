import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity, sequence_length, batch_size, observation_shape, 
                 action_dim, device, oversample_ends=False):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device
        self.oversample_ends = oversample_ends
        self._episodes = deque()
        self._current_episode = []
        self._total_transitions = 0

    def add(self, obs, action, reward, done):
        self._current_episode.append({
            'observation': obs,
            'action': action,
            'reward': reward,
            'done': done
        })

        if done:
            episode = {key: np.array([transition[key] for transition in self._current_episode]) 
                         for key in self._current_episode[0]}
            self._current_episode = []
            self._add_episode(episode)

    def _add_episode(self, episode):
        self._episodes.append(episode)
        self._total_transitions += len(episode['done'])
        while self._total_transitions > self.capacity:
            evicted_episode = self._episodes.popleft()
            self._total_transitions -= len(evicted_episode['done'])

    def sample(self):
        if self._total_transitions < self.sequence_length or len(self._episodes) < 1:
            return None

        batch_obs = np.zeros((self.batch_size, self.sequence_length, *self.observation_shape), dtype=np.uint8)
        batch_actions = np.zeros((self.batch_size, self.sequence_length, self.action_dim), dtype=np.float32)
        batch_rewards = np.zeros((self.batch_size, self.sequence_length), dtype=np.float32)
        batch_dones = np.zeros((self.batch_size, self.sequence_length), dtype=np.bool_)

        for i in range(self.batch_size):
            episode = random.choice(self._episodes)
            max_start_idx = len(episode['done']) - self.sequence_length
            if max_start_idx < 0:
                i -= 1 
                continue

            if self.oversample_ends:
                start_idx = min(random.randint(0, len(episode['done'])), max_start_idx)
            else:
                start_idx = random.randint(0, max_start_idx)

            end_idx = start_idx + self.sequence_length
            
            batch_obs[i] = episode['observation'][start_idx:end_idx]
            action_slice = episode['action'][start_idx:end_idx]
            if action_slice.ndim == 1:
                action_slice = np.expand_dims(action_slice, axis=-1)
            batch_actions[i] = action_slice
            batch_rewards[i] = episode['reward'][start_idx:end_idx]
            batch_dones[i] = episode['done'][start_idx:end_idx]

        return {
            'observations': torch.tensor(batch_obs, dtype=torch.float32, device=self.device) / 255.0 - 0.5,
            'actions': torch.tensor(batch_actions, device=self.device),
            'rewards': torch.tensor(batch_rewards, device=self.device),
            'dones': torch.tensor(batch_dones, dtype=torch.bool, device=self.device)
        }