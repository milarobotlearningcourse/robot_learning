from hw1.roble.infrastructure.replay_buffer import ReplayBuffer
from hw1.roble.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent

class IDMAgent(BaseAgent):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):
        super(IDMAgent, self).__init__()

        # actor/policy

        self._actor = MLPPolicySL(
            **kwargs
        )

        # replay buffer
        self._replay_buffer = ReplayBuffer(self._agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training an IDM agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self._actor.update(ob_no, ac_na)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths):
        self._replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self._replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self._actor.save(path)