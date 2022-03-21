import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import py_trees

from math import exp
import numpy as np

from utils import to_tensor


class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )

        self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                num_options,
                feature_net,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False,
                bt_list = None):

        super(OptionCriticFeatures, self).__init__()
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.feature_net = feature_net
        output_features = feature_net.output_features

        self.Q            = nn.Linear(output_features, num_options)                 # Policy-Over-Options
        if bt_list:
            self.terminations = nn.Linear(output_features, num_options-len(bt_list))                 # Option-Termination
            self.partial_bt = True
            self.num_reg_options = num_options-len(bt_list)
            self.bt_options = len(bt_list)
        else:
            self.terminations = nn.Linear(output_features, num_options) 
            self.partial_bt = False

        self.feature_net.device=device
        self.to(device)
        self.train(not testing)

    def get_state(self, obs,**kwargs):
        return self.feature_net.get_state(obs,**kwargs)

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        if self.partial_bt and current_option>= self.num_reg_options:
            option_termination = True
        else:
            termination = self.terminations(state)[:, current_option].sigmoid()
            option_termination = Bernoulli(termination).sample()
            option_termination = bool(option_termination.item())
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return option_termination, next_option.item()
    
    def get_terminations(self, state):
        terminations = self.terminations(state).sigmoid()
        if self.partial_bt:
            terminations = torch.cat((terminations,torch.ones((terminations.shape[0],self.bt_options)).to(self.device)),dim=1)
        
        return terminations

    def get_action(self, state, option,**kwargs):
        if self.partial_bt and option>= self.num_reg_options:
            self.run_bt(kwargs["bt_list"][option-self.num_reg_options])
            action,logp, entropy = 0,0,0
            return action,logp, entropy
        else:
            return self.feature_net.get_action(state,option,**kwargs)
    
    def run_bt(self,bt_option):
        blackboard = py_trees.blackboard.Client(name="Writer")
        blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
        blackboard.status = py_trees.common.Status.RUNNING
        blackboard.register_key(key="status", access=py_trees.common.Access.READ)
        blackboard.register_key(key="done", access=py_trees.common.Access.READ)
        blackboard.reward = 0
        while blackboard.status == py_trees.common.Status.RUNNING and not blackboard.done:
            bt_option.tick_once()
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model, model_prime, data_batch, args,**kwargs):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(obs,**kwargs).squeeze(0)
    Q      = model.get_Q(states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(next_obs,**kwargs).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime) # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = model.get_state(next_obs,**kwargs).squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args,**kwargs):
    state = model.get_state(obs,**kwargs)
    next_state = model.get_state(next_obs,**kwargs)
    next_state_prime = model_prime.get_state(next_obs,**kwargs)

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    if model.partial_bt and option>= model.num_reg_options:
        termination_loss = 0
    else:
        termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss
