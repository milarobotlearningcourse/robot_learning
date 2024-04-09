# from nose.tools import timed
from numpy.testing import assert_allclose
import numpy as np
import pytest
import sys

import warnings
from hw1.hw1 import my_app
import json
import hydra


# @hydra.main(config_path="hw1/conf", config_name="config")
# https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/

from hydra import compose, initialize

class TestHomeWork1(object):

    @pytest.mark.timeout(600)
    def test_BC(self):
        """
        Check that the BC learned policy can match the expert to 30% expert reward
        """
        ### Load hydra manually.
        initialize(config_path="../../conf", job_name="test_app")
        cfg = compose(config_name="config_hw1", overrides=["+env=absolute_path"])
        cfg.alg.n_iter = 1
        cfg.alg.do_dagger = False
        returns = my_app(cfg)
        # returns = [1.0, 3.0]
        print ("returns: ", returns)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(returns["eval_reward_Average"][-2:]) > 1.0 
        
    @pytest.mark.timeout(600)
    def test_dagger(self):
        """
        Check that the Dagger learned policy can match the expert to 30% expert reward
        """
        ### Load hydra manually.
        # initialize(config_path="../../conf", job_name="test_app")
        cfg = compose(config_name="config_hw1", overrides=["+env=absolute_path"])
        cfg.alg.n_iter = 5
        cfg.alg.do_dagger = True
        returns = my_app(cfg)
        # returns = [1.0, 3.0]
        print ("returns: ", returns)
        # assert np.mean(simData['mean_reward'][-5:]) > -0.5
        assert np.mean(returns["eval_reward_Average"][-2:]) > 3.50 

if __name__ == '__main__':
    pytest.main([__file__])