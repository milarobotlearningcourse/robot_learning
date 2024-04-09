from collections import OrderedDict
import numpy as np
import time

import gym
import torch, pickle
from omegaconf import DictConfig, OmegaConf

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.infrastructure.logging import Logger as TableLogger
from hw1.roble.infrastructure import utils

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params, agent_class=None):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        # self._logger = Logger(self._params['logging']['logdir'])
        self._logger = TableLogger()
        self._logger.add_folder_output(folder_name=self._params['logging']['logdir'])
        self._logger.add_tabular_output(file_name=self._params['logging']['logdir']+"/log_data.csv")
        config_snapshot = {}
        for k, v in self._params.items():
            if k != 'optimizer_spec' \
                and k != 'q_func' \
                    and k != 'env_wrappers' \
                        and k != 'exploration_schedule':
                config_snapshot[k] = v
#        with open(self._params['logging']['logdir']+"/conf.yaml", "w") as fd:
#            fd.write(OmegaConf.to_yaml(OmegaConf.create(config_snapshot)))
#            fd.flush()

        # Set random seeds
        seed = self._params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=self._params['alg']['use_gpu'],
            gpu_id=self._params['alg']['gpu_id']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.create_env(self._params['env']['env_name'], seed)

        # Maximum length for episodes
        # self.params['env']['max_episode_length'] = self._params['env']['max_episode_length'] or self._env.spec.max_episode_steps
        MAX_VIDEO_LEN = self._params['env']['max_episode_length']

        # Is this env continuous, or self._discrete?
        self._params['alg']['discrete'] = isinstance(self._env.action_space, gym.spaces.Discrete)

        # Observation and action sizes
        ob_dim = self._env.observation_space.shape[0]
        ac_dim = self._env.action_space.n if self._params['alg']['discrete'] else self._env.action_space.shape[0]
        self._params['alg']['ac_dim'] = ac_dim
        self._params['alg']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self._env):
            self._fps = 1/self._env.model.opt.timestep
        elif 'env_wrappers' in self._params:
            self._fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self._env.metadata.keys():
            self._fps = self._env.metadata['video.frames_per_second']
        else:
            self._fps = 10

        #############
        ## AGENT
        #############
        ## the **self._params['alg'] is a hack to allow new updates to use kwargs nicely
        combined_params = dict(self._params['alg'].copy())
        combined_params.update(self._params["env"])
        try:
            for key in self._params.keys():
                if "env" in key or "alg" in key:
                    continue
                combined_params[key] = self._params[key]
        except:
            pass
        
        self.add_wrappers()
        self._agent = agent_class(self._env, **combined_params)
        self._log_video = False
        self._log_metrics = True
        
    def create_env(self, env_name, seed):
        import pybullet_envs
        self._env = gym.make(env_name)
        self._eval_env = gym.make(env_name)
        self._env.seed(seed)
        self._eval_env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def set_comet_logger(self, logger):
        self._logger.set_comet_logger(logger)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self._total_envsteps = 0
        self._start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self._params['logging']['video_log_freq'] == 0 and self._params['logging']['video_log_freq'] != -1:
                self._log_video = True
            else:
                self._log_video = False

            # decide if metrics should be logged
            if itr % self._params['logging']['scalar_log_freq'] == 0:
                self._log_metrics = True
            else:
                self._log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self._params['alg']['batch_size']
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self._total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below

            # add collected data to replay buffer
            self._agent.add_to_replay_buffer(paths)

            if self._params['alg']['train_idm']:
                idm_training_logs = self.train_idm()

                # TODO: create a figure from the loss curve in idm_training_logs and add it to your report
                figure = None
                
                # Don't change
                self._agent.reset_replay_buffer()
                self._params['env']['expert_data'] = self._params['env']['expert_unlabelled_data']
                unlabelled_data = self.collect_training_trajectories(
                    itr,
                    initial_expertdata,
                    collect_policy,
                    self._params['alg']['batch_size']
                )
                paths, envsteps_this_batch, train_video_paths = unlabelled_data
                self._agent.use_idm(paths)
                path_list = self._params['env']['expert_unlabelled_data'].split("/")
                path_list.pop(-2)
                path_list[-1] = path_list[-1].replace("unlabelled", "labelled")
                self._params['env']['expert_data'] = "/".join(path_list)
                labelled_data = self.collect_training_trajectories(
                    itr,
                    initial_expertdata,
                    collect_policy,
                    self._params['alg']['batch_size']
                )
                paths, envsteps_this_batch, train_video_paths = labelled_data
                self._agent.reset_replay_buffer()
                self._agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # HW1: implement this function below

            # log/save
            if self._log_video or self._log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self._params['logging']['save_params']:
                    print('\nSaving agent params')
                    self._agent.save('{}/policy_itr_{}.pt'.format(self._params['logging']['logdir'], itr))

        return self._logger.get_table_dict()
    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata=False,
            collect_policy=None,
            batch_size=0,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
            # (1) load the data. In this case you can directly return as follows
            # ``` return loaded_paths, 0, None ```

            # (2) collect `self.params['batch_size']` transitions
        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']

        print("\nCollecting data to be used for training...")

        paths, envsteps_this_batch = TODO
        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN

        train_video_paths = None
        if self._log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self._env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self._params['alg']['num_agent_train_steps_per_iter']):
            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self._params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = TODO

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = TODO
            all_logs.append(train_log)
        return all_logs

    def train_idm(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self._params['alg']['num_idm_train_steps_per_iter']):
            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self._params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = TODO

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train_idm function
            # HINT: keep the agent's training log for debugging
            train_log = TODO
            all_logs.append(train_log)
        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self._env, eval_policy, 
                                                                         self._params['alg']['eval_batch_size'], 
                                                                         self._params['env']['max_episode_length'])

        # save eval rollouts as videos in the video folder (for grading)
        if self._log_video:
            if train_video_paths is not None:
                #save train/eval videos
                print('\nSaving train rollouts as videos...')
                self._logger.log_paths_as_videos(train_video_paths, itr, fps=self._fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self._env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            print('\nSaving eval rollouts as videos...')
            self._logger.log_paths_as_videos(eval_video_paths, itr, fps=self._fps,max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')


        # save eval metrics
        if self._log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            
            # decide what to log
            logs = OrderedDict()

            logs["train_ep_lens"] = train_ep_lens
            logs["eval_ep_lens"] = eval_ep_lens
            logs["train_returns"] = train_returns
            logs["eval_returns"] = eval_returns
            logs["Train_EnvstepsSoFar"] = self._total_envsteps
            logs["TimeSinceStart"] = time.time() - self._start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)
            logs["reward"] = [path["reward"] for path in paths]
            logs["eval_reward"] = [path["reward"] for path in eval_paths]
            for key in paths[0]["infos"][0]:
                logs[str(key)] = [info[key] for path in paths for info in path["infos"]]
                # logs[str(key)] = [value[key] for value in logs[str(key)]]
                logs["eval_"+ str(key)] = [info[key] for path in eval_paths for info in path["infos"]]
            if itr == 0:
                self._initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self._initial_return
            logs["step"] = itr

            for key in logs.keys():
                value = utils.flatten(logs[key])
                self._logger.record_tabular_misc_stat(key, value)
                
            # self._logger.record_tabular_misc_stat("eval_reward", logs["eval_reward"])
            self._logger.dump_tabular()
            print('Done logging...\n\n')
