import numpy as np
import time

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = TODO # HINT: should be the output of resetting the env
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = TODO # HINT: query the policy's get_action function
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        
        # record result of taking that action
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1

        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = TODO # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break
    return Path(obs, image_obs, acs, rewards, next_obs, terminals, infos)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch <= min_timesteps_per_batch:
        TODO
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    TODO
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals, infos):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "infos": infos}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def flatten(matrix):
    ## Check and fix a matrix with different length lists.
    import collections.abc
    if (isinstance(matrix, (collections.abc.Sequence,))  and 
        isinstance(matrix[0], (collections.abc.Sequence, np.ndarray))): ## Flatten possible inhomogeneous arrays
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    else:
        return matrix