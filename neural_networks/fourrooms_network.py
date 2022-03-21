import torch
import torch.nn as nn
from torch.distributions import Categorical


class Fourrooms_net(nn.Module):
    env_name = "fourrooms"
    def __init__(self, env, policies=1,num_actions=4,temp=1, **kwargs):
        
        super(Fourrooms_net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.device=None
        self.output_features = 64
        self.options_W = nn.Parameter(torch.zeros(policies, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(policies, num_actions))
        self.temperature = temp

    def get_state(self, obs,**kwargs):
        from utils import to_tensor
        obs = to_tensor(obs)
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state
    
    def set_device(self,device):
        self.device=device
    
    def get_action(self, state, option,**kwargs):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy