from collections import OrderedDict
import pickle
import os
import sys
import time
import gym
from gym import wrappers
import numpy as np
import torch

from hw2.roble.infrastructure import pytorch_util as ptu
from hw2.roble.infrastructure import utils
from hw3.roble.agents.dqn_agent import DQNAgent
from hw3.roble.agents.ddpg_agent import DDPGAgent
from hw3.roble.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)
from hw2.roble.infrastructure.rl_trainer import RL_Trainer

# how many rollouts to save as videos
MAX_NVIDEO = 1
MAX_VIDEO_LEN = 40 # we overwrite this in the code below

class RL_Trainer(RL_Trainer):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params, agent_class = None):
        
        # Inherit from hw1 RL_Trainer
        register_custom_envs()
        super().__init__(params, agent_class)

        # Get params, create logger
        self._initial_return = 0

        #############
        ## ENV
        #############
        
    def add_wrappers(self):
        # Make the gym environment
        if 'env_wrappers' in self._params:
            # These operations are currently only for Atari envs
            self._env = wrappers.Monitor(
               self._env,
               os.path.join(self._params['logging']['logdir'], "gym"),
               force=True,
               video_callable=(None if self._params['logging']['video_log_freq'] > 0 else False),
            )
            self._env = self._params['env_wrappers'](self._env)
        if 'non_atari_colab_env' in self._params and self._params['logging']['video_log_freq'] > 0:
            self._env = wrappers.Monitor(
                self._env,
                os.path.join(self._params['logging']['logdir'], "gym"),
                force=True,
                video_callable=(None if self._params['logging']['video_log_freq'] > 0 else False),
            )
        self._mean_episode_reward = -float('nan')
        self._best_mean_episode_reward = -float('inf')

        # import plotting (locally if 'obstacles' env)
        if not(self._params['env']['env_name']=='obstacles-roble-v0'):
            import matplotlib
            matplotlib.use('Agg')        

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

        print_period = 1000

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

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
                # while not self._agent.get_replay_buffer().can_sample(self._params['alg']['learning_starts']):
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
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            self._total_envsteps += envsteps_this_batch


            # add collected data to replay buffer
            self._agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()
    

            # log/save
            if ((self._log_video or self._log_metrics) and 
                ( itr % print_period == 0) and 
                (len(all_logs) > 1)):
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self._agent, DQNAgent):
                    self.perform_dqn_logging(itr, all_logs)
                else:
                    self.perform_logging(itr, eval_policy, all_logs)

                if self._params['logging']['save_params']:
                    self._agent.save('{}/agent_itr_{}.pt'.format(self._params['logging']['logdir'], itr))
                    
        results = self._logger.get_table_dict()
        print ("results: ", results)
        return results

    def train_agent(self):
        # TODO: get this from hw1 or hw2
        return all_logs

    ####################################
    ####################################
    def perform_dqn_logging(self, itr, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self._env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self._mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self._best_mean_episode_reward = max(self._best_mean_episode_reward, self._mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self._agent._t
        print("Timestep %d" % (self._agent._t,))
        if self._mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self._mean_episode_reward)
        print("mean reward (100 episodes) %f" % self._mean_episode_reward)
        #if self._best_mean_episode_reward > -5000:
        #    logs["Train_BestReturn"] = np.mean(self._best_mean_episode_reward)
        #print("best mean reward %f" % self._best_mean_episode_reward)
        if self._start_time is not None:
            time_since_start = (time.time() - self._start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start
        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self._logger.record_tabular_misc_stat(key, value)
        
        self._logger.dump_tabular()
        print('Done DQN logging...\n\n')
        # self._logger.flush()
        
    def perform_ddpg_logging(self, itr, all_logs):        
        logs = OrderedDict()
        logs["Train_EnvstepsSoFar"] = self._agent._t
        
        n = 25
        if len(self._agent._rewards) > 0:
            self._mean_episode_reward = np.mean(np.array(self._agent._rewards)[-n:])
            
            logs["Train_AverageReturn"] = self._mean_episode_reward
            logs["Train_CurrentReturn"] = self._agent._rewards[-1]
            
        if len(self._agent._rewards) > n:
            self._best_mean_episode_reward = max(self._best_mean_episode_reward, self._mean_episode_reward)
            
            logs["Train_BestReturn"] = self._best_mean_episode_reward
            
        if len(self._agent._rewards) > 5 * n:   
            self._agent._rewards = self._agent._rewards[n:]
            
        if self._start_time is not None:
            time_since_start = (time.time() - self._start_time)
            logs["TimeSinceStart"] = time_since_start
        
        Q_predictions = []
        Q_targets = []
        policy_actions_mean = []
        policy_actions_std = []
        actor_actions_mean = []
        critic_loss = []
        actor_loss = []
        print_all_logs = True
        for log in all_logs:
            if len(log) > 0:
                print_all_logs = True
                #print(Q_predictions)
                Q_predictions.append(log["Critic"]["Q Predictions"])
                Q_targets.append(log["Critic"]["Q Targets"])
                policy_actions_mean.append(log["Critic"]["Policy Actions"])
                actor_actions_mean.append(log["Critic"]["Actor Actions"])
                critic_loss.append(log["Critic"]["Training Loss"])
                
                if "Actor" in log.keys():
                    actor_loss.append(log["Actor"])
                
        if print_all_logs:
            logs["Q_Predictions"] = Q_predictions
            logs["Q_Targets"] = Q_targets
            logs["Policy_Actions"] = policy_actions_mean
            logs["Actor_Actions"] = actor_actions_mean
            logs["Critic_Loss"] = critic_loss
            
            if len(actor_loss) > 0:
                logs["Actor_Loss"] = actor_loss
            
        for key in logs.keys():
                self._logger.record_tabular_misc_stat(key, logs[key])
                
        # self._logger.record_tabular_misc_stat("eval_reward", logs["eval_reward"])
        self._logger.dump_tabular()
        print('Done DDPG logging...\n\n')

    def perform_logging(self, itr, eval_policy, all_logs):
        
        paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, 
                        collect_policy=eval_policy, 
                        batch_size=self._params['alg']['eval_batch_size'])
                )
        super().perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)
        
        print('Done logging ddpg...\n\n')
        
    def collect_training_trajectories(
            self,
            itr,
            collect_policy,
            batch_size,
            load_initial_expertdata=None
    ):
        return super().collect_training_trajectories(itr + 1, load_initial_expertdata, collect_policy, batch_size)
        