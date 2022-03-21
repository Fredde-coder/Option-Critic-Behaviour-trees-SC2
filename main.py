import numpy as np
import argparse
import torch
import torch.multiprocessing as mp
from copy import deepcopy
import py_trees

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from BT_behaviours.behaviours_OC import *

from experience_replay import ReplayBuffer,TrajectoryBuffer
from utils import make_env, to_tensor
from logger import Logger

import time
import warnings
import logging
import pickle

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='SC2', help='ROM to run')
parser.add_argument('--env-kwargs', default='{"map_name":"BuildMarines", "extract":true,"step_mul":16 , "visualize":false,"get_action_spec":true, "reduced_input":false,"reduce_actions":false}', help='added arguments to environment')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='How many frames to process')
parser.add_argument('--learning-rate',type=float, default=3e-5, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')
parser.add_argument('--bt-comb', type=lambda x: (str(x).lower() == 'true'), default=False, help='Combining with behaviour trees')
parser.add_argument('--partial-bt', type=int, default=0, help='Substituting one or more options for behavior trees(with defaults should be set to 2)')
parser.add_argument('--avoid-bad-env-init', type=lambda x: (str(x).lower() == 'true'), default=True, help='Avoids idle workers, bad rally point and no minerals initalization of environment(also helps initialize other maps)')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default="OCExperiment", help='optional experiment name')
parser.add_argument('--switch-goal', type=lambda x: (str(x).lower() == 'true'), default=False, help='switch goal after 1k eps')
parser.add_argument('--save-model', type=lambda x: (str(x).lower() == 'true'), default=False, help='indicates whether to save the model')
parser.add_argument('--save-checkpoint', type=int, default=50, help='indicates how often to save the model')
parser.add_argument('--load-model', type=str, default=None, help='loads a trained model')#"option_critic_6_180_BT_combTrue"
parser.add_argument('--save-trajectories', type=lambda x: (str(x).lower() == 'true'), default=False, help='saves the episode histories for analysis')
parser.add_argument('--save-frequency', type=int, default=30, help='How often to save trajectories')
parser.add_argument('--eval', type=lambda x: (str(x).lower() == 'true'), default=False, help='Skips training to only evaluate a loaded in model')
parser.add_argument('--test-runs', type=int, default=100, help='Amount of test runs')
parser.add_argument('--max-eps', type=int, default=1500, help='Amount of episodes allowed')
parser.add_argument('--stats-frequency', type=int, default=250, help='How often to save data during episodes(this is to not overload tensorboard with too much data)')


def get_BTs(bt_comb,map_name):
    if bt_comb:
        if map_name=="BuildMarines":
            bt_list = [Build_barrack(),Build_marine(),Build_supply_depot(),Build_scv(),Do_nothing()]
        elif map_name=="DefeatZerglingsAndBanelings":
            bt_list = [Group_attack(control_group=1,enemy_target=105),Group_attack(control_group=1),Group_flee(control_group=1),Group_flee(control_group=1,flee_bot=False)]

        else:
            raise Exception("There are no behavior trees macros defined for the map you are requesting")
    else:
        if map_name=="BuildMarines":
            bt_skill = py_trees.composites.Selector(memory=True)
            b_m = Build_barrack()
            b_s = py_trees.composites.Sequence()
            b_s.add_children( [Check_supply(only_first=True),Build_supply_depot()])
            bt_skill.add_children( [b_m,b_s])
            bt_list=[bt_skill, Build_marine()]
        elif map_name=="DefeatZerglingsAndBanelings":
            bt_list=[Group_attack(control_group=1),Group_flee(control_group=1,flee_bot=False)]
        else:
            raise Exception("There are no partial trees defined for the map you are requesting")
    return bt_list

def map_init_setup(map_name,obs):
    blackboard = py_trees.blackboard.Client(name="Writer")
    if map_name=="BuildMarines":
        from BT_behaviours.behaviours_OC import _UNIT_TYPE,_MINEARL_FIELDS
        unit_type = obs[0].observation["screen"][_UNIT_TYPE]
        mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()
        #Restarts the environment when the no minerals bug occurs
        
        while not mineral_y.any():
            from pysc2.env import sc2_env
            blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
            env = sc2_env.SC2Env(map_name=map_name, visualize=args.env_kwargs["visualize"])
            blackboard.env = env
            obs   = env.reset()
            unit_type = obs[0].observation["screen"][_UNIT_TYPE]
            mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()

        
        blackboard.register_key(key="obs", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
        blackboard.reward = 0
        blackboard.obs = obs[0]
        blackboard.done = False
        obs = obs[0]
        #Since we asssume that all the starting scvs will be active, we need to circumvent such initalizations that make them idle
        #In addition, often such initializations do not have a rally point for the cmd centre onto the minerals therefore we need to set that as well
        idle_stopper = Put_idle_to_work() 
        cmd_centre_rally = Set_worker_rally()
        blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
        blackboard.status = py_trees.common.Status.RUNNING
        blackboard.register_key(key="status", access=py_trees.common.Access.READ)
        while blackboard.status == py_trees.common.Status.RUNNING:
            idle_stopper.tick_once()
        blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
        blackboard.status = py_trees.common.Status.RUNNING
        blackboard.register_key(key="status", access=py_trees.common.Access.READ)
        while blackboard.status == py_trees.common.Status.RUNNING:
            cmd_centre_rally.tick_once()
    else:
        blackboard.register_key(key="obs", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
        blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
        blackboard.reward = 0
        blackboard.obs = obs[0]
        blackboard.done = False
        
    



def run(args):
    if args.bt_comb or args.partial_bt:
        bt_list = get_BTs(args.bt_comb,args.env_kwargs["map_name"])
        env, is_atari,feature_net = make_env(args.env,policies=args.num_options,bt_list=bt_list, partial_bt = args.partial_bt,temperature=args.temp,**args.env_kwargs)
        blackboard = py_trees.blackboard.Client(name="Writer")
        blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
        if "get_action_spec" in args.env_kwargs:
            blackboard.env = env[0]
        else:
            blackboard.env = env
    else:
        env, is_atari,feature_net = make_env(args.env,policies=args.num_options,temperature=args.temp,**args.env_kwargs)
        
        blackboard = py_trees.blackboard.Client(name="Writer")
        blackboard.register_key(key="env", access=py_trees.common.Access.WRITE)
        if "get_action_spec" in args.env_kwargs:
            blackboard.env = env[0]
        else:
            blackboard.env = env
        
        bt_list = None
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    if "get_action_spec" in args.env_kwargs:
        env, action_spec = env
    
    if args.partial_bt:
        option_critic = option_critic(
            num_options=args.num_options,
            feature_net=feature_net,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device,
            testing=args.eval,
            bt_list=bt_list
        )
    else:
        option_critic = option_critic(
            num_options=args.num_options,
            feature_net=feature_net,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device,
            testing=args.eval
        )
    if args.load_model:
        checkpoint = torch.load("models/"+args.load_model)
        option_critic.load_state_dict(checkpoint["model_params"])
        option_critic.eps_start = args.epsilon_min

    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
    #print(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "env_seed" in args.env_kwargs:
        env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    if args.save_trajectories:
        episode_history = TrajectoryBuffer(capacity=args.max_history, seed=args.seed)
        ep_histdict_positive = {}
        ep_histdict_negative = {}
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}--{args.env_kwargs['map_name']}")
    

    steps = 0 ;
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total and logger.n_eps<args.max_eps:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}
        if args.eval and (logger.n_eps+1)%args.test_runs==0:
            break

        obs   = env.reset()
        if args.avoid_bad_env_init:
            map_init_setup(args.env_kwargs["map_name"],obs)
            blackboard.register_key(key="obs", access=py_trees.common.Access.READ)
            if args.bt_comb:
                obs = blackboard.obs
            else:
                obs = [blackboard.obs]

        state = option_critic.get_state(obs)
        greedy_option  = option_critic.greedy_option(state) if not "extract" in args.env_kwargs else option_critic.greedy_option(state[0])
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        'models/option_critic_{args.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 1300:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        'models/option_critic_{args.seed}_1300')
            break
        if args.save_model and (logger.n_eps+1) % args.save_checkpoint == 0:
            if args.load_model:
                eps = int(args.load_model.split("_")[-1])+1
                torch.save({'model_params': option_critic.state_dict()},
                        f'models/option_critic_{args.num_options}_{logger.n_eps+eps+1}_{args.env_kwargs["map_name"]}')
            else:
                if args.bt_comb:
                    torch.save({'model_params': option_critic.state_dict()},
                                f'models/option_critic_{args.num_options}_{logger.n_eps+1}_BT_comb_{args.env_kwargs["map_name"]}')
                elif args.partial_bt:
                    torch.save({'model_params': option_critic.state_dict()},
                                f'models/option_critic_{args.num_options}_{logger.n_eps+1}_partial_bt_{args.env_kwargs["map_name"]}')
                else:
                    torch.save({'model_params': option_critic.state_dict()},
                                f'models/option_critic_{args.num_options}_{logger.n_eps+1}_{args.env_kwargs["map_name"]}')

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0;reward=0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
                if (args.partial_bt and current_option>= option_critic.num_reg_options):
                    blackboard = py_trees.blackboard.Client(name="Writer")
                    blackboard.register_key(key="obs", access=py_trees.common.Access.WRITE)
                    blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
                    blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
                    blackboard.reward = 0
                    blackboard.obs = obs[0]
                    blackboard.done = False
                      
            action, logp, entropy = option_critic.get_action(state, current_option) if not "get_action_spec" in args.env_kwargs else option_critic.get_action(state, current_option,obs=obs[0], action_spec = action_spec,bt_list=bt_list) 
           
            if args.bt_comb or (args.partial_bt and current_option>= option_critic.num_reg_options):
                blackboard = py_trees.blackboard.Client(name="Reader")
                blackboard.register_key(key="obs", access=py_trees.common.Access.READ)
                blackboard.register_key(key="reward", access=py_trees.common.Access.READ)
                blackboard.register_key(key="done", access=py_trees.common.Access.READ)
                next_obs,reward, done = blackboard.obs,blackboard.reward,blackboard.done
                if args.partial_bt:
                    buffer.push(obs[0][3], current_option, reward, next_obs[3], done)
                    next_obs = [next_obs]

                else:
                    buffer.push(obs[3], current_option, reward, next_obs[3], done)

            elif not "extract" in args.env_kwargs:
                next_obs, reward, done, _ = env.step(action) 
                buffer.push(obs, current_option, reward, next_obs, done)
            else:
                next_obs = env.step(action)
                reward, done = option_critic.feature_net.extract_obs_info(next_obs)
             
                buffer.push(obs[0].observation, current_option, reward, next_obs[0].observation, done)
                if args.save_trajectories:
                    episode_history.push(obs[0].observation['player'], current_option, reward,next_obs[0].observation["player"], next_obs[0].observation["available_actions"], action)

            old_state = state
            
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size and not args.eval:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, args,return_nonspatial=True)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args,batch=True,return_nonspatial=True)
                    loss += critic_loss
                if not (args.partial_bt and current_option>= option_critic.num_reg_options) or steps % args.update_frequency == 0:
                    optim.zero_grad()
                    loss.backward()
        
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(next_obs)
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option) if not "extract" in args.env_kwargs else option_critic.predict_option_termination(state[0], current_option)

            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            if args.stats_frequency and (steps-1)%args.stats_frequency==0:
                if  (args.partial_bt and current_option>= option_critic.num_reg_options):
                    logger.log_data(steps, actor_loss, critic_loss, entropy, epsilon)
                else:
                    logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        
        if args.env=="SC2":
            print(f"> ep {logger.n_eps} done. total_steps={steps} | reward={rewards} | episode_steps={ep_steps} "\
                f"| hours={(time.time()-logger.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        
        if args.save_trajectories:  
            if rewards>0:
                ep_histdict_positive[str(logger.n_eps)] = np.array(episode_history.retrive_all())

                
            else:
                ep_histdict_negative[str(logger.n_eps)] = np.array(episode_history.retrive_all())
                
            
            
            if logger.n_eps%args.save_frequency==0:
                if len(ep_histdict_positive)>0:
                    with open(f"episode_histories/trajectories_positive_{logger.n_eps}.npz","wb+") as queue_save_file:
                        np.savez_compressed(queue_save_file,**ep_histdict_positive)
                if len(ep_histdict_negative)>0:
                        with open(f"episode_histories/trajectories_negative_{logger.n_eps}.npz","wb+") as queue_save_file:
                            np.savez_compressed(queue_save_file,**ep_histdict_negative)

            episode_history.clear()
    


if __name__=="__main__":
    args = parser.parse_args()
    try:
        import json
        args.env_kwargs = json.loads(args.env_kwargs)
    except Exception as e:
        args.env_kwargs = {}
    

    run(args)
