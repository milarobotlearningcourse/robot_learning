import numpy as np
from gymnasium.core import Wrapper


class ActionRepeatWrapper(Wrapper):
    def __init__(self, env, max_repeat):
        super().__init__(env)
        self.max_repeat= max_repeat

        assert max_repeat >= 1

    def _sample_num_repeat(self):
        return int(np.random.randint(1, self.max_repeat))

    def step(self, action):
        # TODO: repeat action a random number of times
        return super(ActionRepeatWrapper, self).step(action)
