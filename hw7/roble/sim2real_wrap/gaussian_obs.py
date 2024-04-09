import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType


class GaussianObsWrapper(ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        return self._noise_obs(observation)

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def _noise_obs(self, obs):
        # TODO: add noise
        return obs
