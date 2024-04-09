from hw1.roble.infrastructure.utils import *


class ReplayBuffer(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, max_size=1000000):

        # store each rollout
        self._paths = []

        # store (concatenated) component arrays from each rollout
        self._obs = None
        self._acs = None
        self._rews = None
        self._next_obs = None
        self._terminals = None

    def __len__(self):
        if self._obs:
            return self._obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self._paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self._obs is None:
            self._obs = observations[-self._max_size:]
            self._acs = actions[-self._max_size:]
            self._rews = rewards[-self._max_size:]
            self._next_obs = next_observations[-self._max_size:]
            self._terminals = terminals[-self._max_size:]
        else:
            self._obs = np.concatenate([self._obs, observations])[-self._max_size:]
            self._acs = np.concatenate([self._acs, actions])[-self._max_size:]
            if concat_rew:
                self._rews = np.concatenate(
                    [self._rews, rewards]
                )[-self._max_size:]
            else:
                if isinstance(rewards, list):
                    self._rews += rewards
                else:
                    self._rews.append(rewards)
                self._rews = self._rews[-self._max_size:]
            self._next_obs = np.concatenate(
                [self._next_obs, next_observations]
            )[-self._max_size:]
            self._terminals = np.concatenate(
                [self._terminals, terminals]
            )[-self._max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        assert (
                self._obs.shape[0]
                == self._acs.shape[0]
                == self._rews.shape[0]
                == self._next_obs.shape[0]
                == self._terminals.shape[0]
        )
        ## TODO return batch_size number of random entries from each of the 5 component arrays above
        ## HINT 1: use np.random.permutation to sample random indices
        ## HINT 2: return corresponding data points from each array (i.e., not different indices from each array)
        ## HINT 3: look at the sample_recent_data function below
        raise NotImplementedError #return TODO, TODO, TODO, TODO, TODO

    def sample_recent_data(self, batch_size=1):
        return (
            self._obs[-batch_size:],
            self._acs[-batch_size:],
            self._rews[-batch_size:],
            self._next_obs[-batch_size:],
            self._terminals[-batch_size:],
        )
