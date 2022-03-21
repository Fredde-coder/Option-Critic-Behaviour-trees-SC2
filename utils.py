import gym
import numpy as np
import torch
import types 

from gym.wrappers import AtariPreprocessing, TransformReward
from gym.wrappers import FrameStack as FrameStack_

from environments.fourrooms import Fourrooms
from neural_networks import *
from pysc2.env import sc2_env

var_dict = globals().copy()
try:
    net_list = [eval(name+"."+name[0].upper()+name[1:-4]) for name, var in var_dict.items() if "network" in name and isinstance(var,types.ModuleType)]
except Exception as e:
    raise(Exception(e+"\nOne of the networks does not conform to the convention of being named network and the class being\
        capitalized and having the same name as the file but ending with net instead of network"))



class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

def make_env(env_name, **kwargs):

    if env_name == 'fourrooms':
        env = Fourrooms()
        return env, False, get_env_net(env_name=env_name,env=env,**kwargs)
    elif env_name == "atari":
        env = gym.make(env_name)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
            env = TransformReward(env, lambda r: np.clip(r, -1, 1))
            env = FrameStack(env, 4)
        return env, is_atari, get_env_net(env_name=env_name)
    elif env_name=="SC2":
        from absl import flags
        FLAGS = flags.FLAGS
        FLAGS([""])
        print('Initializing temporary environment to retrive action_spec')
        action_spec = sc2_env.SC2Env(map_name=kwargs["map_name"], visualize=False).action_spec()
        feature_net = get_env_net(env_name,action_spec,**kwargs)
        
        print('Initializing training environment')
        real_env = sc2_env.SC2Env(map_name=kwargs["map_name"], visualize=kwargs["visualize"])
        return [real_env,action_spec], False, feature_net


def get_env_net(env_name,env,**kwargs):
    for net in net_list:
        if net.env_name==env_name:
            return net(env,**kwargs)
    raise Exception("There is no feature net for that environment")

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs
