import gymnasium
import numpy as np
from gym import Wrapper

class LastActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        low, high = env.observation_space.low, env.observation_space.high
        low = np.concatenate((low, env.action_space.low))
        high = np.concatenate((high, env.action_space.high))

        self.observation_space = gymnasium.spaces.Box(low=low, high=high)

    def _make_observation(self, obs, last_act):
        # TODO: concatenate obs and last_act
        pass


    def reset(self, **kwargs):
        ret = super(LastActionWrapper, self).reset(**kwargs)
        return self._make_observation(ret[0], self.action_space.sample() * 0), ret[1]

    def step(self, action):
        ret = super(LastActionWrapper, self).step(action)
        return self._make_observation(ret[0], action), *ret[1:]
