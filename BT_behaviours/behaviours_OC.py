from numpy.lib.function_base import append
import py_trees
from py_trees import blackboard
from pysc2.lib import actions
from pysc2.env import environment
from pysc2.lib import features
import numpy as np
#Structured info
_PLAYER_ID = 0
_MINERALS = 1
_VESPENE = 2
_FOOD_USED = 3
_FOOD_CAP = 4
_FOOD_USED_BY_ARMY = 5
_FOOD_USED_BY_WORKERS = 6
_IDLE_WORKER_COUNT = 7
_ARMY_COUNT = 8
_WARP_GATE_COUNT = 9
_LARVA_COUNT = 10
_ZERGLING = 105
_BANELING = 9 
_PLAYER_HOSTILE = 4



# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_RALLY_WORKERS_SCREEN = actions.FUNCTIONS.Rally_Workers_screen.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id



# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_SELECTED = features.SCREEN_FEATURES.selected.index
_SUPPLY_USED = 3
_SUPPLY_CAP = 4
_SUPPLY_ARMY_USE = 5
# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_MARINE = 48
_TERRAN_BARRACKS = 21
_TERRAN_SUPPLY_DEPOT = 19
_MINEARL_FIELDS = 341

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


class Put_idle_to_work(py_trees.behaviour.Behaviour):
    def __init__(self,name="Put_idle_to_work"):
        super(Put_idle_to_work, self).__init__(name)
        

    def initialise(self):
        
        self.logger.debug("  %s [Put_idle_to_work::initialise()]" % self.name)

    def update(self):
      env = get_env()
      obs = get_obs()
      self.logger.debug("  %s [Put_idle_to_work::update()]" % self.name)
      if _SELECT_IDLE_WORKER in obs.observation["available_actions"]:
        #unit_type = obs.observation["screen"][_UNIT_TYPE]
        #unit_y, unit_x = (unit_type == 0).nonzero()
        #unit_y = np.random.choice(unit_y)
        #unit_x = np.random.choice(unit_x)  
        #target = [unit_x,unit_y]
        a =  actions.FunctionCall(_SELECT_IDLE_WORKER, [_SELECT_ALL])
        obs = env.step([a])
        status = py_trees.common.Status.RUNNING
        update_obs(obs,status)
        obs = get_obs()
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()
        #target = [ int(mineral_x.mean()),int(mineral_y.mean())]
        if mineral_y.any() and _HARVEST_GATHER_SCREEN in obs.observation["available_actions"]:
          target = [ mineral_x[0],mineral_y[0]+1]
          a = actions.FunctionCall(_HARVEST_GATHER_SCREEN, [_NOT_QUEUED, target])
          obs = env.step([a])
          status = py_trees.common.Status.SUCCESS
          update_obs(obs,status)
          return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status
        
      else:
        a= actions.FunctionCall(_NOOP, [])
        obs = env.step([a])
        status = py_trees.common.Status.FAILURE
        update_obs(obs,status)
        return status

class Set_worker_rally(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Set_worker_rally"):
        super(Set_worker_rally, self).__init__(name)
    
    def initialise(self):
        
        self.logger.debug("  %s [Set_worker_rally::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Set_worker_rally::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not _TERRAN_COMMANDCENTER in obs.observation["single_select"]:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
          
          target = [int(unit_x.mean()), int(unit_y.mean())]
          
          a= actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
          status = py_trees.common.Status.RUNNING
          obs = env.step([a])
          update_obs(obs,status)
          return status
        elif _RALLY_WORKERS_SCREEN in obs.observation["available_actions"]:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()
          if mineral_y.any():
            target = [mineral_x[0],mineral_y[0]+1]
            a = actions.FunctionCall(_RALLY_WORKERS_SCREEN, [_NOT_QUEUED,target])
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status)
            return status
          else:
            a= actions.FunctionCall(_NOOP, [])
            obs = env.step([a])
            status = py_trees.common.Status.FAILURE
            update_obs(obs,status)
            return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status


class Build_barrack(py_trees.behaviour.Behaviour):

    def __init__(self,name="Build barrack"):
        super(Build_barrack, self).__init__(name)
        self.barracks_queued = False

    def initialise(self):
        
        self.logger.debug("  %s [Build_barrack::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Build_barrack::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not _TERRAN_SCV in obs.observation["single_select"]:
          self.barracks_queued = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
          
          target = [unit_x[0], unit_y[0]]
          
      
          a= actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
          obs = env.step([a])
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status)
          return status
        elif _BUILD_BARRACKS in obs.observation["available_actions"] and not self.barracks_queued:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == 0).nonzero()
          unit_y = np.random.choice(unit_y)
          unit_x = np.random.choice(unit_x)
          
          target = [unit_x,unit_y]
          
          self.barracks_queued = True
          
          a= actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
          obs = env.step([a])
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status)
          return status
        
        elif _HARVEST_GATHER_SCREEN in obs.observation["available_actions"] and self.barracks_queued:
          self.barracks_queued = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()
          if mineral_y.any():
            target = [ mineral_x[0],mineral_y[0]+1]
            a = actions.FunctionCall(_HARVEST_GATHER_SCREEN, [_QUEUED, target])
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status)
            return status
          else:
            a= actions.FunctionCall(_NOOP, [])
            obs = env.step([a])
            status = py_trees.common.Status.FAILURE
            update_obs(obs,status)
            return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status


class Build_marine(py_trees.behaviour.Behaviour):
    def __init__(self,name="Build marine"):
        super(Build_marine, self).__init__(name)
        

    def initialise(self):
        
        self.logger.debug("  %s [Build_marine::initialise()]" % self.name)

    def update(self):
      env = get_env()
      obs = get_obs()
      self.logger.debug("  %s [Build_marine::update()]" % self.name)
      if not _TERRAN_BARRACKS in obs.observation["single_select"] and not _TERRAN_BARRACKS in obs.observation["multi_select"]:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        if unit_y.any():
          unit_y = np.random.choice(unit_y)
          unit_x = np.random.choice(unit_x)  
          target = [unit_x,unit_y]
          a =  actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
          obs = env.step([a])
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status)
          return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status
      elif  _TRAIN_MARINE in obs.observation["available_actions"]:
        a = actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        obs = env.step([a])
        status = py_trees.common.Status.SUCCESS
        update_obs(obs,status)
        return status
      else:
        a= actions.FunctionCall(_NOOP, [])
        obs = env.step([a])
        status = py_trees.common.Status.FAILURE
        update_obs(obs,status)
        return status

class Build_supply_depot(py_trees.behaviour.Behaviour):

    def __init__(self,name="Build supply depot"):
        super(Build_supply_depot, self).__init__(name)
        self.supply_depot_queued = False
    
    def initialise(self):
        
        self.logger.debug("  %s [Build_supply_depot::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Build_supply_depot::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not _TERRAN_SCV in obs.observation["single_select"]:
          self.supply_depot_queued = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
          
          target = [unit_x[0], unit_y[0]]
          
          self.scv_selected = True
      
          a= actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
          obs = env.step([a])
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status)
          return status
        elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"] and not self.supply_depot_queued: 
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          #unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
          unit_y, unit_x = (unit_type == 0).nonzero()
          unit_y = np.random.choice(unit_y)
          unit_x = np.random.choice(unit_x)
          
          target = [unit_x,unit_y]
          self.supply_depot_queued = True
          
          
          a = actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
          
          obs = env.step([a])
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status)
          return status
        elif _HARVEST_GATHER_SCREEN in obs.observation["available_actions"] and self.supply_depot_queued:
          self.supply_depot_queued = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          mineral_y, mineral_x = (unit_type ==  _MINEARL_FIELDS).nonzero()
          if mineral_y.any():
            target = [ mineral_x[0],mineral_y[0]+1]
            a = actions.FunctionCall(_HARVEST_GATHER_SCREEN, [_QUEUED, target])
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status)
            return status
          else:
            a= actions.FunctionCall(_NOOP, [])
            obs = env.step([a])
            status = py_trees.common.Status.FAILURE
            update_obs(obs,status)
            return status
        
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status

class Build_scv(py_trees.behaviour.Behaviour):

    def __init__(self,name="Build scv"):
        super(Build_scv, self).__init__(name)
    
    def initialise(self):
        
        self.logger.debug("  %s [Build_scv::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Build_scv::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not _TERRAN_COMMANDCENTER in obs.observation["single_select"]:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
          
          target = [int(unit_x.mean()), int(unit_y.mean())]
          
          a= actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
          status = py_trees.common.Status.RUNNING
          obs = env.step([a])
          update_obs(obs,status)
          return status
        elif _TRAIN_SCV in obs.observation["available_actions"]:
          a = actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
          obs = env.step([a])
          status = py_trees.common.Status.SUCCESS
          update_obs(obs,status)
          return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status)
          return status

class Do_nothing(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Do nothing"):
        super(Do_nothing, self).__init__(name)
    
    def initialise(self):
        
        self.logger.debug("  %s [Do_nothing::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Do_nothing::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        a= actions.FunctionCall(_NOOP, [])
        obs = env.step([a])
        status = py_trees.common.Status.SUCCESS
        update_obs(obs,status, action=[a])
        return status


  

class Action_node(py_trees.behaviour.Behaviour):
    
    def __init__(self,action_id=_NOOP,action_arg="rand", arg_values=[]):
        try:
          name = str(actions.FUNCTIONS[action_id]).split()[0].split("/")[1]
        except Exception:
          name = "Does not exist"
        super(Action_node, self).__init__(name)
        self.action_id = action_id
        self.action_arg = action_arg
        self.arg_values = arg_values
    
    def initialise(self):
        
        self.logger.debug("  %s [Action_node::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Action_node::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if self.action_id in obs.observation["available_actions"]:
            if self.action_arg=="rand":
                arg_values = []
                for v in self.arg_values:
                  arg_value = [np.random.choice(v)]
                  if v==4096:
                      arg_value = np.unravel_index(arg_value[0],(64,64))
                  arg_values.append(arg_value)
            a = actions.FunctionCall(self.action_id,arg_values)
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status, action=[a])
            return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status, action=[a])
          return status

class Check_node(py_trees.behaviour.Behaviour):
    
    def __init__(self,action_id,states=[],abstract_layer="player",thresholding =True):
        try:
          name ="state_check_"+ str(actions.FUNCTIONS[action_id]).split()[0].split("/")[1]
        except Exception:
          name = "Does not exist"
        super(Check_node, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.states = states
        self.abstract_layer = abstract_layer
        self.thresholding = thresholding
        self.thresholds = [{f"min":states[:,i].min(), f"max":states[:,i].max()} for i in range(states.shape[1])]
      
    def return_thresholds(self):
        return self.thresholds
    
    def initialise(self):
        
        self.logger.debug("  %s [Check_node::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Check_node::update()]" % self.name)
        obs = get_obs()
        abstract_state = obs.observation[self.abstract_layer]
        if self.thresholding:
          for i,v in enumerate(abstract_state):
            if i == _SUPPLY_ARMY_USE:
              continue
            if v>=self.thresholds[i]["min"] and v<=self.thresholds[i]["max"]:
              continue

            return py_trees.common.Status.FAILURE
            
        return py_trees.common.Status.SUCCESS
            


class Check_barracks(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Check barracks",first_barrack=False):
        super(Check_barracks, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.first_barrack = first_barrack
    
    def initialise(self):
        
        self.logger.debug("  %s [Check_barracks::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Check_barracks::update()]" % self.name)
        obs = get_obs()
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        if not unit_y.any() and not self.first_barrack:
          return py_trees.common.Status.SUCCESS
        elif 0==obs.observation["player"][_SUPPLY_CAP]-obs.observation["player"][_SUPPLY_USED]:
          return py_trees.common.Status.FAILURE
        elif 0==obs.observation["player"][_SUPPLY_ARMY_USE] and self.first_barrack:
          return py_trees.common.Status.FAILURE
        else:
          return py_trees.common.Status.SUCCESS

class Check_marines(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Check marines"):
        super(Check_marines, self).__init__(name)
    
    def initialise(self):
        
        self.logger.debug("  %s [Check_marines::initialise()]" % self.name)

    def update(self):
        """
        When is this called?
          Every time your behaviour is ticked.

        What to do here?
          - Triggering, checking, monitoring. Anything...but do not block!
          - Set a feedback message
          - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        """
        self.logger.debug("  %s [Check_marines::update()]" % self.name)
        return py_trees.common.Status.SUCCESS

class Check_supply(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Check supply",only_first=False):
        super(Check_supply, self).__init__(name)
        self.only_first = only_first

    def initialise(self):
        
        self.logger.debug("  %s [Check_supply::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Check_supply::update()]" % self.name)
        obs = get_obs()
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        if not unit_y.any():
          return py_trees.common.Status.SUCCESS
        elif self.only_first:
          return py_trees.common.Status.FAILURE
        if 5>obs.observation["player"][_SUPPLY_CAP]-obs.observation["player"][_SUPPLY_USED]:
          return py_trees.common.Status.SUCCESS
        else:
          return py_trees.common.Status.FAILURE


def get_env():
    blackboard = py_trees.blackboard.Client(name="Reader")
    blackboard.register_key(key="env", access=py_trees.common.Access.READ)
    return blackboard.env

def get_obs():
    blackboard = py_trees.blackboard.Client(name="Reader")
    blackboard.register_key(key="obs", access=py_trees.common.Access.READ)
    return blackboard.obs

def update_obs(obs,status, action=None):
    blackboard = py_trees.blackboard.Client(name="Writer")
    blackboard.register_key(key="obs", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="reward", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="done", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="status", access=py_trees.common.Access.WRITE)
    if action:
      blackboard.register_key(key="action", access=py_trees.common.Access.WRITE)
      blackboard.action = action
    blackboard.obs = obs[0]
    observation = obs[0]
    # is episode over?
    done = observation.step_type==environment.StepType.LAST
    reward = observation.reward
    blackboard.reward += reward
    blackboard.done = done
    blackboard.status = status
  

def make_control_groups(obs,env):
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == _MARINE).nonzero()
    
    
    top_corner = [min(unit_x), min(unit_y)]
    bottom_corner = [max(unit_x), max(unit_y)]
    middle = [unit_x[0]+(unit_x[-1]-unit_x[0]), unit_y[0]+(unit_y[-1]-unit_y[0])/2]
    e = actions.FUNCTIONS[_SELECT_CONTROL_GROUP].args 

    a= actions.FunctionCall(_SELECT_RECT, [_NOT_QUEUED, top_corner,middle])
    obs = env.step([a])
    a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[1],[1]])
    obs = env.step([a])
    middle[1]+=3
    a= actions.FunctionCall(_SELECT_RECT, [_NOT_QUEUED, middle,bottom_corner])
    obs = env.step([a])
    a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[1],[2]])
    obs = env.step([a])
    status = py_trees.common.Status.SUCCESS
    update_obs(obs,status)
    return status
    '''
    a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0],[1]])
    obs = env.step([a])

    unit_y, unit_x = (unit_type == _ZERGLING).nonzero()
    

    target = [unit_x[0], unit_y[0]]
    e = actions.FUNCTIONS[_ATTACK_SCREEN].args 
    
    a = actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED,target])
    obs = env.step([a])
    e = actions.FUNCTIONS[_MOVE_SCREEN].args 
    if abs(top_corner[0]-unit_x[0])<abs(bottom_corner[0]-unit_x[0]):
      a = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,[63,63]])
      enemy_left = True
    else:
      a = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,[0,0]])
      enemy_right = True
    obs = env.step([a])


    a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0],[2]])
    obs = env.step([a])
    unit_y, unit_x = (unit_type == _BANELING).nonzero()
    target = [unit_x[0], unit_y[0]]

    a = actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED,target])
    obs = env.step([a])
    '''

class Group_attack(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Group_attack",control_group=0, enemy_target = _BANELING):
        super(Group_attack, self).__init__(name)
        self.control_group = control_group
        self.control_group_selected = False
        self.enemy_target = enemy_target
    
    def initialise(self):
        
        self.logger.debug("  %s [Group_attack::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Group_attack::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not self.control_group_selected:
          a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0],[self.control_group]])
          obs = env.step([a])
          self.control_group_selected = True
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status, action=[a])
          return status
        elif self.control_group_selected and _ATTACK_SCREEN in obs.observation["available_actions"]:
          self.control_group_selected = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == self.enemy_target).nonzero()
          if unit_y.any():
            target = [unit_x[0], unit_y[0]]
            
            a = actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED,target])
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status, action=[a])
            return status
          else:
            a= actions.FunctionCall(_NOOP, [])
            obs = env.step([a])
            status = py_trees.common.Status.FAILURE
            update_obs(obs,status, action=[a])
            return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status, action=[a])
          return status

class Group_flee(py_trees.behaviour.Behaviour):
    
    def __init__(self,name="Group_flee",control_group=0, flee_bot = True, enemy_target = _BANELING):
        super(Group_flee, self).__init__(name)
        self.control_group = control_group
        self.control_group_selected = False
        self.flee_bot = flee_bot
        self.enemy_target = enemy_target
    
    def initialise(self):
        
        self.logger.debug("  %s [Group_flee::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Group_flee::update()]" % self.name)
        env = get_env()
        obs = get_obs()
        if not self.control_group_selected:
          a = actions.FunctionCall(_SELECT_CONTROL_GROUP, [[0],[self.control_group]])
          obs = env.step([a])
          self.control_group_selected = True
          status = py_trees.common.Status.RUNNING
          update_obs(obs,status, action=[a])
          return status
        elif self.control_group_selected and _MOVE_SCREEN in obs.observation["available_actions"]:
          self.control_group_selected = False
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          unit_y, unit_x = (unit_type == _MARINE).nonzero()
    
          top_corner = [min(unit_x), min(unit_y)]
          bottom_corner = [max(unit_x), max(unit_y)]

          enemy_y, enemy_x = (unit_type == self.enemy_target).nonzero()
          if enemy_y.any():
            if abs(top_corner[0]-enemy_x[0])<abs(bottom_corner[0]-enemy_x[0]):
              a = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,[63,63*self.flee_bot]])
              enemy_left = True
            else:
              a = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED,[0,63*self.flee_bot]])
              enemy_right = True
            obs = env.step([a])
            status = py_trees.common.Status.SUCCESS
            update_obs(obs,status, action=[a])
            return status
          else:
            a= actions.FunctionCall(_NOOP, [])
            obs = env.step([a])
            status = py_trees.common.Status.FAILURE
            update_obs(obs,status, action=[a])
            return status
        else:
          a= actions.FunctionCall(_NOOP, [])
          obs = env.step([a])
          status = py_trees.common.Status.FAILURE
          update_obs(obs,status, action=[a])
          return status
