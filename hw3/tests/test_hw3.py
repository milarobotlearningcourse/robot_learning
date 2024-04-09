from numpy.testing import assert_allclose
import numpy as np
import pytest
import json

import hydra, json
from omegaconf import DictConfig, OmegaConf

from hw3.hw3 import my_app
from hydra import compose, initialize

class TestTD3(object):

    # @pytest.mark.timeout(600)
    def test_td3(self):
        
        # assert 1.0 > -0.5
        initialize(config_path="../../conf", job_name="test_app")
        cfg = compose(config_name="config_hw3", overrides=[])
        cfg.alg.rl_alg="td3"
        cfg.env.exp_name='test_td3'
        cfg.env.env_name="HalfCheetah-v2"
        cfg.env.atari=False
        cfg.logging.video_log_freq=-1 
        cfg.logging.scalar_log_freq=1000
        cfg.alg.n_iter=3000
        cfg.alg.learning_starts=1000
        result = my_app(cfg)
        
        print ("returns: ", result)
        
        # assert np.mean(result['trainer/Train_AverageReturn'][-5:]) > -2.0
        assert np.mean(result['eval_reward_Average']) > -20.0       
        
        
    # @pytest.mark.timeout(600)
    def test_ddpg(self):
        
        # assert 1.0 > -0.5
        initialize(config_path="../../conf", job_name="test_app")
        cfg = compose(config_name="config_hw3", overrides=[])
        cfg.alg.rl_alg="ddpg"
        cfg.env.exp_name='test_ddpg'
        cfg.env.env_name="HalfCheetah-v2"
        cfg.env.atari=False
        cfg.logging.video_log_freq=-1 
        cfg.logging.scalar_log_freq=1000
        cfg.alg.n_iter=3000
        cfg.alg.learning_starts=1000
        result = my_app(cfg)
        
        print ("returns: ", result)
        
        # assert np.mean(result['trainer/Train_AverageReturn'][-5:]) > -2.0
        assert np.mean(result['eval_reward_Average']) > -20.0   
        

    # @pytest.mark.timeout(600)
    def test_sac(self):
        
        # assert 1.0 > -0.5
        initialize(config_path="../../conf", job_name="test_app")
        cfg = compose(config_name="config_hw3", overrides=[])
        cfg.alg.rl_alg="sac"
        cfg.env.exp_name='test_ddpg'
        cfg.env.env_name="HalfCheetah-v2"
        cfg.env.atari=False
        cfg.logging.video_log_freq=-1 
        cfg.logging.scalar_log_freq=1000
        cfg.alg.n_iter=3000
        cfg.alg.learning_starts=1000
        result = my_app(cfg)
        
        print ("returns: ", result)
        
        # assert np.mean(result['trainer/Train_AverageReturn'][-5:]) > -2.0
        assert np.mean(result['eval_reward_Average']) > -20.0
            

if __name__ == '__main__':
    nose.main([__file__])