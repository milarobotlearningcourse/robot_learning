import numpy as np
import copy

from hw3.roble.policies.MLP_policy import MLPPolicyStochastic
from hw3.roble.critics.sac_critic import SACCritic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class SACAgent(DDPGAgent):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)
        
        self._actor = MLPPolicyStochastic(
            **kwargs
        )

        self._q_fun = SACCritic(self._actor, 
                               **kwargs)
        
    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self._last_obs must always point to the new latest observation.
        """        

        # TODO: Take the code from DDPG Agent and make sure the remove the exploration noise
        return