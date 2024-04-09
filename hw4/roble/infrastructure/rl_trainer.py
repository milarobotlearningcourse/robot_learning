from collections import OrderedDict
import pickle
import os
import sys
import time
import gym
from gym import wrappers
import numpy as np
import torch

from hw3.roble.infrastructure.rl_trainer import RL_Trainer
from hw2.roble.infrastructure import pytorch_util as ptu
from hw2.roble.infrastructure import utils
from hw3.roble.agents.dqn_agent import DQNAgent
from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.agents.td3_agent import TD3Agent
from hw3.roble.agents.sac_agent import SACAgent
from hw4.roble.agents.pg_agent import PGAgent

from hw3.roble.infrastructure.dqn_utils import (
        get_wrapper_by_name
)
from hw4.roble.envs.ant.create_maze_env import create_maze_env
from hw4.roble.envs.reacher.reacher_env import create_reacher_env
from hw4.roble.infrastructure.gclr_wrapper import GoalConditionedEnv, GoalConditionedEnvV2
from hw4.roble.infrastructure.hrl_wrapper import HRLWrapper
# how many rollouts to save as videos
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(RL_Trainer):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params, agent_class = None):

        #############
        ## INIT
        #############
        # Inherit from hw1 RL_Trainer
        super().__init__(params, agent_class)
        

        # Get params, create logger
        self._params = params

        # Set random seeds
        seed = self._params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        if 'env_wrappers' in self._params:
            # These operations are currently only for Atari envs
            self._env = wrappers.Monitor(
                self._env,
                os.path.join(self._params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self._params['logging']['video_log_freq'] > 0 else False),
            )
            self._env = self._params['env_wrappers'](self._env)
            self._mean_episode_reward = -float('nan')
            self._best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self._params and self._params['logging']['video_log_freq'] > 0:
            self._env = wrappers.Monitor(
                self._env,
                os.path.join(self._params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self._params['logging']['video_log_freq'] > 0 else False),
            )
            self._mean_episode_reward = -float('nan')
            self._best_mean_episode_reward = -float('inf')

        self._env.seed(seed)


        #############
        ## AGENT
        #############

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
            # print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self._params['logging']['video_log_freq'] == 0 and self._params['logging']['video_log_freq'] != -1:
                self._log_video = True
            else:
                self._log_video = False

            # decide if metrics should be logged
            if self._params['logging']['scalar_log_freq'] == -1:
                self._log_metrics = False
            elif itr % self._params['logging']['scalar_log_freq'] == 0:
                self._log_metrics = True
            else:
                self._log_metrics = False

            # collect trajectories, to be used for training
            if isinstance(self._agent, DQNAgent) or isinstance(self._agent, DDPGAgent):
                # only perform an env step and add to replay buffer for DQN and DDPG
                self._agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self._params['alg']['batch_size']
                if itr==0:
                    use_batchsize = self._params['alg']['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, collect_policy, use_batchsize)
                )

            self._total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            
            # print("collected ", len(paths[0]), " trajectories")
            self._agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # print("\nTraining agent...")
            all_logs = self.train_agent()
    

            # log/save
            if ((self._log_video or self._log_metrics) and 
                (len(all_logs) >= 1)):
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self._agent, DQNAgent):
                    self.perform_dqn_logging(itr, all_logs)
                else:
                    self.perform_logging(itr, eval_policy, all_logs)

                if self._params['logging']['save_params']:
                    self._agent.save('{}/agent_itr_{}.pt'.format(self._params['logging']['logdir'], itr))
            if  self._params['alg']['on_policy']:       
                self._agent.clear_mem()


    ####################################
    ####################################
        
    def create_env(self, env_name, seed):
        import pybullet_envs
        if self._params['env']['env_name'] == 'antmaze':
            self._env = create_maze_env('AntMaze')
        elif self._params['env']['env_name'] == 'reacher':
            self._env = create_reacher_env()
        elif self._params['env']['env_name'] == 'widowx':
            from hw4.roble.envs.roboverse.widowx import create_widow_env
            self._env = create_widow_env(observation_mode='state')
        else:
            self._env = gym.make(env_name)
            
                # Call your goal conditioned wrapper here (You can modify arguments depending on your implementation)
        if self._params['env']['task_name'] == 'gcrl':
            self._env = GoalConditionedEnv(self._env, **self._params['env'])     
        elif self._params['env']['task_name'] == 'gcrl_v2':
            self._env = GoalConditionedEnvV2(self._env, self._params['env'])
        elif self._params['env']['task_name'] == 'hrl':
            self._env = HRLWrapper(self._env, self._params['env'])
        else:
            pass
        
        self._env.seed(seed)
        # self._eval_env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

