import numpy as np
import copy

from hw3.roble.policies.MLP_policy import MLPPolicyDeterministic
from hw3.roble.critics.td3_critic import TD3Critic
from hw3.roble.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        super().__init__(env, **kwargs)
        
        self._q_fun = TD3Critic(self._actor, 
                               **kwargs)
        