import collections

import gymnasium
import numpy as np
from gym import Wrapper


class HistoryWrapper(Wrapper):
    def __init__(self, env, length):
        super().__init__(env)
        self.length = length

        low, high = env.observation_space.low, env.observation_space.high
        low = np.array([[low] * length]).squeeze().flatten()
        high = np.array([[high] * length]).squeeze().flatten()

        self.observation_space = gymnasium.spaces.Box(low=low,
                                                      high=high)
        self._reset_buf()

    def _reset_buf(self):
        # TODO: reset history buffer
        pass

    def _make_observation(self):

        # TODO: concatenate history into obs
        pass


    def reset(self, **kwargs):

        self._reset_buf()
        ret = super(HistoryWrapper, self).reset(**kwargs)

        # TODO: add first state to history buffer
        pass
        return self._make_observation(), ret[1]


    def step(self, action):
        ret = super(HistoryWrapper, self).step(action)

        # TODO: add state to history buffer
        pass

        return self._make_observation(), *ret[1:]
