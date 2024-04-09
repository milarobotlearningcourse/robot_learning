# adds sim2real wrappers
from typing import SupportsFloat

import numpy as np
from gymnasium import RewardWrapper

from hw7.roble.sim2real_wrap.gaussian_act import GaussianActWrapper
from hw7.roble.sim2real_wrap.gaussian_obs import GaussianObsWrapper
from hw7.roble.sim2real_wrap.history import HistoryWrapper
from hw7.roble.sim2real_wrap.last_action import LastActionWrapper
from hw7.roble.sim2real_wrap.noop_reset import RandomActResetEnv
from hw7.roble.sim2real_wrap.repeat_act import ActionRepeatWrapper

thunk_sim2real_wrap = None

def make_thunk(cfg):
    global thunk_sim2real_wrap

    def sim2real_wrap(env):
        if cfg.max_action_repeat_on_reset > 1:
            # done first so as not to influence the History and the LastAction wrappers
            env = RandomActResetEnv(env, cfg.max_action_repeat_on_reset)

        if cfg.history_len > 1:
            env = HistoryWrapper(env, length=cfg.history_len) # todo
        if cfg.add_last_action:
            env = LastActionWrapper(env)

        if cfg.gaussian_obs_scale > 0:
            env = GaussianObsWrapper(env, scale=cfg.gaussian_obs_scale)
        if cfg.gaussian_act_scale > 0:
            env = GaussianActWrapper(env, scale=cfg.gaussian_act_scale)

        if cfg.action_repeat_max > 1:
            env = ActionRepeatWrapper(env, cfg.action_repeat_max)

        return env

    thunk_sim2real_wrap = sim2real_wrap
    return thunk_sim2real_wrap