import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ObsType, WrapperObsType, ActionWrapper, WrapperActType, ActType


class GaussianActWrapper(ActionWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def action(self, action: WrapperActType) -> ActType:
        return self._noise_act(action)

    def _noise_act(self, act):
        # TODO: add noise
        return act
