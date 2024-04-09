import numpy as np

class HRLWrapper(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, base_env, low_level_policy=None, high_level_policy=None, **kwargs):
        self.base_env = base_env
        # TODO
        ## Load the policy \pi(a|s,g,\theta^{low}) you trained from Q4.
        ## Make sure the policy you load was trained with the same goal_frequency
        pass

    def reset_step_counter(self):
        # TODO
        pass

    def step(self, a):
        ## Add code to compute a new goal-conditioned reward
        # TODO
        sub_goal = action # The high level policy action \pi(g|s,\theta^{hi}) is the low level goal.
        for range(goal_frequency):
            ## Get the action to apply in the environment
            ## HINT you need to use \pi(a|s,g,\theta^{low})
            ## Step the environment
            pass ## Remove this
        
        # return s_{t+k}, r_{t+k}, done, info
        pass

    def create_state(self, obs, goal):
        return np.concatenate([obs, goal])

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_spaces
