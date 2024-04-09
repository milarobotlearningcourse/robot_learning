# hw 2 imports
from hw2.roble.infrastructure.utils import *
from hw1.roble.infrastructure.replay_buffer import ReplayBuffer as ReplayBuffer1

class ReplayBuffer(ReplayBuffer1):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, max_size=1000000):
        super(ReplayBuffer, self).__init__(max_size=max_size)

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        for path in paths:
            self._paths.append(path)

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, concatenated_rews, next_observations, terminals = convert_listofrollouts(paths)

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self._obs is None:
            self._obs = observations[-self._max_size:]
            self._acs = actions[-self._max_size:]
            self._next_obs = next_observations[-self._max_size:]
            self._terminals = terminals[-self._max_size:]
            self._concatenated_rews = concatenated_rews[-self._max_size:]
        else:
            self._obs = np.concatenate([self._obs, observations])[-self._max_size:]
            self._acs = np.concatenate([self._acs, actions])[-self._max_size:]
            self._next_obs = np.concatenate(
                [self._next_obs, next_observations]
            )[-self._max_size:]
            self._terminals = np.concatenate(
                [self._terminals, terminals]
            )[-self._max_size:]
            self._concatenated_rews = np.concatenate(
                [self._concatenated_rews, concatenated_rews]
            )[-self._max_size:]

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self._paths))[:num_rollouts]
        return self._paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self._paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self._obs.shape[0] == self._acs.shape[0] == self._concatenated_rews.shape[0] == self._next_obs.shape[0] == self._terminals.shape[0]
        rand_indices = np.random.permutation(self._obs.shape[0])[:batch_size]
        return self._obs[rand_indices], self._acs[rand_indices], self._concatenated_rews[rand_indices], self._next_obs[rand_indices], self._terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return self._obs[-batch_size:], self._acs[-batch_size:], self._concatenated_rews[-batch_size:], self._next_obs[-batch_size:], self._terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self._paths[index]
                index -=1
                num_recent_rollouts_to_return +=1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self._paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews, unconcatenated_rews = convert_listofrollouts(rollouts_to_return)
            return observations, actions, unconcatenated_rews, next_observations, terminals
        