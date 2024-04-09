
import numpy as np 
from gym.spaces import Box

    
class GoalConditionedEnv(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        # TODO
        self._env = base_env

    def success_fn(self,last_reward):
        # TODO
        pass
    
    def reset(self):
        # Add code to generate a goal from a distribution
        # TODO
        pass

    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        info["reached_goal"] = self.success_fn(reward)
        pass

    def create_state(self,obs,goal):
        ## Add the goal to the state
        # TODO
        pass
    
    @property
    def action_space(self):
        return self._env.action_space
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def metadata(self):
        return self._env.metadata
    @property
    def unwrapped(self):
        return self._env


class GoalConditionedEnvV2(GoalConditionedEnv):

    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, **kwargs):
        super().__init__(base_env, **kwargs)
        # TODO
        super().__init__(base_env, **kwargs)
        self._env = base_env

    def reset(self):
        # Add code to generate a goal from a distribution
        # TODO
        pass
        
    def success_fn(self,last_reward):
        # TODO
        pass
        
    def reset_step_counter(self):
        # Add code to track how long the current goal has been used.
        # TODO
        pass

    def step(self, a):
        ## Add code to control the agent for a number of timesteps and 
        ## change goals after k timesteps.
        # TODO
        pass