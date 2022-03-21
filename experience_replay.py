import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size,**kwargs):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        if "torch" in kwargs:
            return torch.stack(obs), option, reward, torch.stack(next_obs), done
        else:
            return np.stack(obs), option, reward, np.stack(next_obs), done
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class TrajectoryBuffer(object):
    def __init__(self, capacity, seed=42):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, option, reward, next_obs,next_obs_actions, action):
        self.buffer.append((obs, option, reward, next_obs,next_obs_actions, action))
    
    def retrive_all(self):
        episode_list = []
        for i in range(len(self.buffer)):
            episode_list.append(self.buffer[i])
        return episode_list
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

        
