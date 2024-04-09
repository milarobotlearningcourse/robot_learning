import abc
import itertools
import numpy as np
import torch
import hw1.roble.util.class_util as classu

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy
from hw1.roble.policies.MLP_policy import MLPPolicy
from hw2.roble.infrastructure.utils import normalize
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

class MLPPolicyPG(MLPPolicy):
    
    @classu.hidden_member_initialize
    def __init__(self, 
                 *args,
                 **kwargs):

        super().__init__(
                *args,
                 **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        action_distribution = self(observations)
        loss = - action_distribution.log_prob(actions) * advantages
        loss = loss.mean()
    
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        train_log = {
            'Training_Loss': ptu.to_numpy(loss),
        }
        if self._nn_baseline:
            targets_n = normalize(q_values, np.mean(q_values), np.std(q_values))
            targets_n = ptu.from_numpy(targets_n)
            # targets_n = ptu.from_numpy(q_values * ((1.0-self._gamma)/1.0))
            baseline_predictions = self._baseline(observations).squeeze()
            assert baseline_predictions.dim() == baseline_predictions.dim()
    
            for i in range(self._num_critic_updates_per_agent_update):
                baseline_predictions = self._baseline(observations).squeeze()
                baseline_loss = F.mse_loss(baseline_predictions, targets_n)
                self._baseline_optimizer.zero_grad()
                baseline_loss.backward()
                self._baseline_optimizer.step()
                train_log["Critic_Loss"] = ptu.to_numpy(baseline_loss)
        else:
            train_log["Critic_Loss"] = ptu.to_numpy(0)

        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self._baseline(observations)
        return ptu.to_numpy(pred.squeeze())