import numpy as np

from hw3.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw3.roble.policies.argmax_policy import ArgMaxPolicy
from hw3.roble.critics.dqn_critic import DQNCritic

class DQNAgent(object):
    
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):

        self.env = env
        self.last_obs = self.env.reset()

        self.num_actions = self.env.action_space.n
        self.replay_buffer_idx = None
        
        self.critic = DQNCritic(**kwargs)
        self._actor = ArgMaxPolicy(self.critic)

        lander = kwargs["env_name"].startswith("LunarLander")
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            kwargs['replay_buffer_size'], kwargs['frame_history_len'], lander=lander)
        self._t = 0
        self.num_param_updates = 0
        
        self.exploration = kwargs["exploration_schedule"]
        self.batch_size = kwargs["train_batch_size"]
        self.learning_starts = kwargs["learning_starts"]
        self.learning_freq = kwargs["learning_freq"]
        self.target_update_freq = kwargs["target_update_freq"]
        
    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        self.replay_buffer_idx = -1

        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = TODO
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action = TODO
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            action = TODO
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        TODO

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        TODO

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        TODO

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self._t > self.learning_starts
                and self._t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # TODO fill in the call to the update function using the appropriate tensors
            log = self.critic.update(
                TODO
            )

            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                TODO

            self.num_param_updates += 1
        self._t += 1
        return log
