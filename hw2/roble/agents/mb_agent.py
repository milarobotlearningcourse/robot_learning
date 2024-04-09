
# hw2 imports
from hw2.roble.models.ff_model import FFModel
from hw2.roble.infrastructure.replay_buffer import ReplayBuffer
from hw2.roble.policies.MPC_policy import MPCPolicy

# hw1 imports
from hw1.roble.agents.base_agent import BaseAgent
from hw1.roble.infrastructure.utils import *

import numpy as np


class MBAgent(BaseAgent):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, **kwargs):
        super(MBAgent, self).__init__()

        self._env = env.unwrapped

        self._dyn_models = []
        for i in range(self._ensemble_size):
            model = FFModel(
                **kwargs
            )
            self._dyn_models.append(model)

        self._actor = MPCPolicy(
            self._env,
            dyn_models=self._dyn_models,
            **kwargs
        )

        self._replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self._ensemble_size)

        for i in range(self._ensemble_size):
            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful

            # observations = # TODO(Q1)
            # actions = # TODO(Q1)
            # next_observations = # TODO(Q1)

            # # use datapoints to update one of the dyn_models
            # model =  # TODO(Q1)
            log = model.update(observations, actions, next_observations,
                                self._data_statistics)
            loss = log['Training Loss']
            losses.append(loss)

        avg_loss = np.mean(losses)
        return {
            'Training Loss': avg_loss,
        }

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self._replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self._data_statistics = {
            'obs_mean': np.mean(self._replay_buffer._obs, axis=0),
            'obs_std': np.std(self._replay_buffer._obs, axis=0),
            'acs_mean': np.mean(self._replay_buffer._acs, axis=0),
            'acs_std': np.std(self._replay_buffer._acs, axis=0),
            'delta_mean': np.mean(
                self._replay_buffer._next_obs - self._replay_buffer._obs, axis=0),
            'delta_std': np.std(
                self._replay_buffer._next_obs - self._replay_buffer._obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self._actor._data_statistics = self._data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self._replay_buffer.sample_random_data(
            batch_size * self._ensemble_size)
        
    def save(self, path):
        print("NOTE: Nothing to save for MB agent (maybe the models?)")
        return