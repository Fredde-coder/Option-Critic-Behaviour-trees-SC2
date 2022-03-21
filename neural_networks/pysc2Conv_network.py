from functools import reduce
import torch
import torch.nn as nn
from torch.nn.modules import activation, flatten
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import numpy as np
import copy

from s2clientprotocol.sc2api_pb2 import Observation
from pysc2.lib import actions
from pysc2.env import environment
import py_trees

 #Architecture is FullyConv agent from deepminds article on pysc2, taken from x repository
class Pysc2Conv_net(nn.Module):
    env_name = "SC2"
    def __init__(self,action_spec, policies=1,bt_list=None, partial_bt=0, get_action_spec=True,device='cuda',reduced_input=False,reduce_actions=False,**kwargs):
        
        super(Pysc2Conv_net, self).__init__()
       
        self.device = device
        self.lstm = True if "lstm" in kwargs and kwargs["lstm"] else False
        self.partial_bt = partial_bt
        nonspatial_size = 12
        minimap_channels = 7
        screen_channels = 17
        if reduced_input:
            self.reduced_input=True
            self.minimap_ind = 4
            self.screen_ind = 14
            minimap_channels = 1
            screen_channels = 1
        else:
            self.reduced_input = False
        if reduce_actions:
            self.reduce_actions=True
            self.allowed_actions = ["no_op","select_rect","Move_screen"]
        else:
            self.reduce_actions = False

        
        
        

        self.screen_features = nn.Sequential(
            nn.Conv2d(screen_channels,16,\
                kernel_size=[5,5], stride=(1,1),padding="same"),
            nn.ReLU(),
            nn.Conv2d(16,32,\
                kernel_size=[3,3], stride=(1,1),padding="same"),
            nn.ReLU(),
        )
        self.minimap_features = nn.Sequential(
            nn.Conv2d(minimap_channels,16,\
                kernel_size=[5,5], stride=(1,1),padding="same"),
            nn.ReLU(),
            nn.Conv2d(16,32,\
                kernel_size=[3,3], stride=(1,1),padding="same"),
            nn.ReLU(),
        )
        self.nonspatial_dense = nn.Sequential(
            nn.Linear(nonspatial_size, 32),
            nn.Tanh(),
        )
        self.to(device)

        self.latent_nonspatial = nn.Sequential(
                nn.Linear(32+64**3, 256),
                nn.ReLU()
            )

        if self.lstm:
            self.hidden = (torch.zeros((1,1,256)).to(device).detach(), torch.zeros((1,1,256)).to(device).detach())
            self.latent_nonspatial_lstm = nn.LSTM(256,256,num_layers=1)
            
        self.output_features = 256 #For higher level abstraction layers
        
        if bt_list and not self.partial_bt:
            self.policy_base_actions = nn.ModuleList([nn.Sequential(
                nn.Linear(256, len(bt_list)),
                nn.Softmax(dim=1)
                )\
                for i in range(policies)])
            self.macro_actions = True
        else:
            if self.partial_bt:
                policies = policies - self.partial_bt
            self.policy_base_actions = nn.ModuleList([nn.Sequential(
                nn.Linear(256, len(action_spec.functions)),
                nn.Softmax(dim=1)
                )\
                for i in range(policies)])
            self.macro_actions = False

            spatial_arguments = ['screen', 'minimap', 'screen2']
            self.policy_arg_nonspatial = nn.ModuleDict()
            for arg in action_spec.types:
                if arg.name not in spatial_arguments:
                    self.policy_arg_nonspatial[arg.name] = nn.ModuleDict()
                    for dim, size in enumerate(arg.sizes):
                        self.policy_arg_nonspatial[arg.name][str(dim)] = nn.Sequential(
                            nn.Linear(256, size),
                            nn.Softmax(dim=1)
                        )
            self.policy_arg_nonspatial = nn.ModuleList([copy.deepcopy(self.policy_arg_nonspatial) for i in range(policies)])

            self.latent_spatial = nn.ModuleDict()
            for arg in spatial_arguments:
                self.latent_spatial[arg] = nn.Conv2d(64, 1,\
                                kernel_size=[1,1],stride=(1,1), padding="same")
            self.latent_spatial = nn.ModuleList([copy.deepcopy(self.latent_spatial) for i in range(policies)])
        self.to(device)
            

    def extract_obs_info(self,obs):
        observation = obs[0]
        # is episode over?
        done = observation.step_type==environment.StepType.LAST
        reward = observation.reward
        return reward, done


    @torch.no_grad()
    def process_obs(self,observation, minmax_norm=False,standardization=False,standard_nonspatial=False,**kwargs):
        if "batch" in kwargs and kwargs["batch"]:
            reward=0
            done = False
            features = dict()
            for key in observation[0].keys():
                features[key] = np.array([obs[key] for obs in observation])
            nonspatial_stack = []
            nonspatial_stack = np.log(features['player'] + 1.)
            nonspatial_stack = np.concatenate((nonspatial_stack, features['game_loop']),1)
            # spatial_minimap features
            minimap_stack = (features['minimap'])
            # spatial_screen features
            screen_stack = (features['screen'])
            if self.reduced_input:
                minimap_stack = (np.expand_dims(features['minimap'][:,self.minimap_ind,:,:],axis=1))
                # spatial_screen features
                screen_stack = (np.expand_dims(features['screen'][:,self.screen_ind,:,:],axis=1))
            if standardization and not minmax_norm:
                nonspatial_stack = np.array([(nonspatial_stack[i]-nonspatial_stack[i].mean())/nonspatial_stack[i].std() if nonspatial_stack[i].std()!=0 else \
                    nonspatial_stack[i]-nonspatial_stack[i].mean()\
                    for i in range(nonspatial_stack.shape[0])])

                minimap_stack = np.array([[(minimap_stack[j,i,:,:]-minimap_stack[j,i,:,:].mean())/minimap_stack[j,i,:,:].std() if minimap_stack[j,i,:,:].std()!=0 else \
                    minimap_stack[j,i,:,:]-minimap_stack[j,i,:,:].mean()\
                    for i in range(minimap_stack.shape[1])] for j in range(minimap_stack.shape[0])])

                screen_stack = np.array([[(screen_stack[j,i,:,:]-screen_stack[j,i,:,:].mean())/screen_stack[j,i,:,:].std() if screen_stack[j,i,:,:].std()!=0 else \
                    screen_stack[j,i,:,:]-screen_stack[j,i,:,:].mean()\
                    for i in range(screen_stack.shape[1])] for j in range(screen_stack.shape[0])])
            
            elif minmax_norm:
                nonspatial_stack = torch.stack([(nonspatial_stack[i]-nonspatial_stack[i].min())/(nonspatial_stack[i].max()-nonspatial_stack[i].min()) if (nonspatial_stack[i].max()-nonspatial_stack[i].min())!=0 else \
                    nonspatial_stack[i] for i in range(nonspatial_stack.shape[0])]).to(self.device)

                minimap_stack = torch.stack([torch.stack([(minimap_stack[j,i]-minimap_stack[j,i].min())/(minimap_stack[j,i].max()-minimap_stack[j,i].min()) if (minimap_stack[j,i].max()-minimap_stack[j,i].min())!=0 else \
                    minimap_stack[j,i] for i in range(minimap_stack.shape[1])]) for j in range(minimap_stack.shape[0])]).to(self.device)

                screen_stack = torch.stack([torch.stack([(screen_stack[j,i]-screen_stack[j,i].min())/(screen_stack[j,i].max()-screen_stack[j,i].min()) if (screen_stack[j,i].max()-screen_stack[j,i].min())!=0 else \
                    screen_stack[j,i] for i in range(screen_stack.shape[1])]) for j in range(screen_stack.shape[0])]).to(self.device)
            
            elif standard_nonspatial:
                nonspatial_stack = np.array([(nonspatial_stack[i]-nonspatial_stack[i].mean())/nonspatial_stack[i].std() if nonspatial_stack[i].std()!=0 else \
                    nonspatial_stack[i]-nonspatial_stack[i].mean()\
                    for i in range(nonspatial_stack.shape[0])])

            nonspatial_stack = torch.from_numpy(nonspatial_stack).float().to(self.device)
            minimap_stack = torch.from_numpy(minimap_stack).float().to(self.device)
            screen_stack = torch.from_numpy(screen_stack).float().to(self.device)
            
            return nonspatial_stack,minimap_stack,screen_stack, reward, done
        else:
            if not self.macro_actions:
                observation = observation[0]
            # is episode over?
            done = observation.step_type==environment.StepType.LAST
            reward = observation.reward
            # features
            features = observation.observation
            # nonspatial features
            nonspatial_stack = []
            nonspatial_stack = np.log(features['player'].reshape(-1) + 1.)
            nonspatial_stack = np.concatenate((nonspatial_stack, features['game_loop'].reshape(-1)))
            # spatial_minimap features
            minimap_stack = (features['minimap'])
            # spatial_screen features
            screen_stack = (features['screen'])
            if self.reduced_input:
                
                minimap_stack = (np.expand_dims(features['minimap'][self.minimap_ind],axis=0))
                # spatial_screen features
                screen_stack = (np.expand_dims(features['screen'][self.screen_ind],axis=0))
            if standardization and not minmax_norm:
                nonspatial_stack = (nonspatial_stack-nonspatial_stack.mean())/nonspatial_stack.std() if nonspatial_stack.std()!=0 else nonspatial_stack-nonspatial_stack.mean()
                
                minimap_stack = np.array([(minimap_stack[i]-minimap_stack[i].mean())/minimap_stack[i].std() if minimap_stack[i].std()!=0 else \
                    minimap_stack[i]-minimap_stack[i].mean() for i in range(minimap_stack.shape[0])])

                screen_stack = np.array([(screen_stack[i]-screen_stack[i].mean())/screen_stack[i].std() if screen_stack[i].std()!=0 else \
                    screen_stack[i]-screen_stack[i].mean() for i in range(screen_stack.shape[0])])
            elif minmax_norm:
                nonspatial_stack = ((nonspatial_stack-nonspatial_stack.min())/(nonspatial_stack.max()-nonspatial_stack.min()) if nonspatial_stack.max()-nonspatial_stack.min()!=0\
                else nonspatial_stack)

                minimap_stack = torch.stack([(minimap_stack[i]-minimap_stack[i].min())/(minimap_stack[i].max()-minimap_stack[i].min()) if (minimap_stack[i].max()-minimap_stack[i].min())!=0 else \
                    minimap_stack[i] for i in range(minimap_stack.shape[0])])

                screen_stack = torch.stack([(screen_stack[i]-screen_stack[i].min())/(screen_stack[i].max()-screen_stack[i].min()) if (screen_stack[i].max()-screen_stack[i].min())!=0 else \
                    screen_stack[i] for i in range(screen_stack.shape[0])])
            elif standard_nonspatial:
                nonspatial_stack = (nonspatial_stack-nonspatial_stack.mean())/nonspatial_stack.std() if nonspatial_stack.std()!=0 else nonspatial_stack-nonspatial_stack.mean()

            
            nonspatial_stack = torch.tensor(np.expand_dims(nonspatial_stack, axis=0),dtype=torch.float,device=self.device)#.float().to(self.device)

            minimap_stack = torch.tensor(np.expand_dims(minimap_stack, axis=0),dtype=torch.float,device=self.device)#.float().to(self.device)
            
            screen_stack = torch.tensor(np.expand_dims(screen_stack, axis=0),dtype=torch.float,device=self.device)#.float().to(self.device)

            if "return_norm" in kwargs and kwargs["return_norm"]:
                return (nonspatial_stack.squeeze(0).detach(),minimap_stack.squeeze(0).detach(),screen_stack.squeeze(0).detach()), reward, done
            else:
                return nonspatial_stack,minimap_stack,screen_stack, reward, done
    
    def get_state(self,obs,**kwargs):
        if "processed" in kwargs and kwargs["processed"]:
            nonspatial_stack,minimap_stack,screen_stack = obs
            reward = 0
            done = False
        else:
            nonspatial_stack,minimap_stack,screen_stack, reward, done = self.process_obs(obs,**kwargs)

        output_screen_features = self.screen_features(screen_stack)
        output_minimap_features = self.minimap_features(minimap_stack)
        output_nonspatial_dense = self.nonspatial_dense(nonspatial_stack)

        batch = 1 if "batch" in kwargs and kwargs["batch"] else 0
        
        flattened_nonspatial =torch.flatten(output_nonspatial_dense,start_dim=batch)
        flattened_screen = torch.flatten(output_screen_features,start_dim=batch)
        flattened_minimap = torch.flatten(output_minimap_features,start_dim=batch)
        nonspatial_input = torch.cat((flattened_nonspatial,flattened_screen,flattened_minimap),dim=batch)
        nonspatial_input = torch.unsqueeze(nonspatial_input,0)
        nonspatial_latent_output = self.latent_nonspatial(nonspatial_input)

        if self.lstm:
            nonspatial_latent_output = nonspatial_latent_output.unsqueeze(0)
            self.latent_nonspatial_lstm.flatten_parameters()
            nonspatial_latent_output, hidden = self.latent_nonspatial_lstm(nonspatial_latent_output,(self.hidden))
            if "lstm_save" in kwargs and not kwargs["lstm_save"]:
                pass
            else:
                self.hidden = tuple([h.data for h in hidden])
            nonspatial_latent_output = nonspatial_latent_output.squeeze(0)
            
        if "squeeze" in kwargs and kwargs["squeeze"]:
            nonspatial_latent_output = nonspatial_latent_output.squeeze(0)

        latent_spatial = torch.cat((output_screen_features,output_minimap_features),dim=1)
        
        if "extract" in kwargs and kwargs["extract"]:
            if "return_norm" in kwargs and kwargs["return_norm"]:
                return (nonspatial_latent_output, latent_spatial), reward, done, (nonspatial_stack.squeeze(0),minimap_stack.squeeze(0),screen_stack.squeeze(0))
            else:
                return (nonspatial_latent_output, latent_spatial), reward, done
            
        if "return_plus" in kwargs and kwargs["return_plus"]:
            return (nonspatial_latent_output, latent_spatial), (nonspatial_stack.squeeze(0).detach(),minimap_stack.squeeze(0).detach(),screen_stack.squeeze(0).detach())
        else:
            if "return_nonspatial" in kwargs:
                return nonspatial_latent_output
            else:
                return nonspatial_latent_output, latent_spatial

    def get_action(self,state, policy,obs=None, action_spec={},squeezed=False,**kwargs):

        probs = []
        logp=0
        entropy=0
        nonspatial_latent_output,latent_spatial = state
        action_logits = self.policy_base_actions[policy](nonspatial_latent_output)
        if self.macro_actions:
            action_dist = Categorical(action_logits)
            action_id = action_dist.sample()
            logp = logp +action_dist.log_prob(action_id)
            entropy = entropy+ action_dist.entropy()
            blackboard = py_trees.blackboard.Client(name="Writer")
            blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
            blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
            blackboard.status = py_trees.common.Status.RUNNING
            blackboard.register_key(key="status", access=py_trees.common.Access.READ)
            blackboard.register_key(key="done", access=py_trees.common.Access.READ)
            blackboard.reward = 0
            while blackboard.status == py_trees.common.Status.RUNNING and not blackboard.done:
                kwargs["bt_list"][action_id].tick_once()
            return action_id, logp, entropy
        else:
            available_actions = []
            for action_id, action in enumerate(action_logits[0]):
                if action_id in obs.observation['available_actions']:
                    if self.reduce_actions:
                        action_name = actions.FUNCTIONS[action_id].name
                        if action_name in self.allowed_actions:
                            available_actions.append(action_id)
                    else:
                        available_actions.append(action_id)
            if squeezed:
                action_logits = action_logits[available_actions]
            else:
                action_logits = action_logits[:,available_actions]
            
            action_logits = action_logits+ 1e-20
            base_action_prob = action_logits/torch.sum(action_logits)
            action_dist = Categorical(base_action_prob)
            action_id = action_dist.sample()
            logp = logp +action_dist.log_prob(action_id)
            entropy = entropy+ action_dist.entropy()
            action_id = available_actions[action_id.item()]
            
            if "return_probs" in kwargs and kwargs["return_probs"]:
                    action_id = kwargs["action_id"]
            probs.append({available_actions[i]:prob for i,prob in enumerate(base_action_prob[0])})


                        
            arguments = []
            spatial_arguments = ['screen', 'minimap', 'screen2']
            

            for argument in action_spec.functions[action_id].args:
                name = argument.name
                if name not in spatial_arguments:
                    argument_value = []
                    for dim, size in enumerate(argument.sizes):
                        arg_prob = self.policy_arg_nonspatial[policy][name][str(dim)](nonspatial_latent_output)
                        arg_dist = Categorical(arg_prob)
                        sampled_arg = arg_dist.sample()
                        logp = logp +arg_dist.log_prob(sampled_arg)
                        entropy = entropy+ arg_dist.entropy()
                        argument_value.append(sampled_arg.item())
                        probs.append(arg_prob)
                else:
                    out = self.latent_spatial[policy][name](latent_spatial)
                    arg_prob = torch.nn.Softmax(dim=2)(out.view(1,1,-1))
                    arg_dist = Categorical(arg_prob)
                    sampled_arg = arg_dist.sample()
                    logp = logp +arg_dist.log_prob(sampled_arg)
                    entropy = entropy+ arg_dist.entropy()
                    argument_value = np.unravel_index(sampled_arg.item(), out.shape[2:])
                    probs.append(arg_prob)
                arguments.append(argument_value)

            if "return_probs" in kwargs and kwargs["return_probs"]:
                return probs
            a = actions.FunctionCall(action_id, arguments)
            return [a], logp, entropy
        


        