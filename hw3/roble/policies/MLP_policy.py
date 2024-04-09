import abc
import itertools
import numpy as np
import torch

from hw2.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy 
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.1
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # DONE return the action that the policy prescribes
        obs = ptu.from_numpy(observation)
        actions = None

        if self._deterministic:
            return ptu.to_numpy(self(obs))

        if self._discrete:
            distrib = self(obs)
            return ptu.to_numpy(distrib.sample())

        return ptu.to_numpy(self(obs).rsample())

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if self._deterministic:
                action_distribution = self._mean_net(observation)
            else:
                
                batch_mean = self._mean_net(observation)
                # scale_tril = torch.diag(torch.exp(self._logstd))
                scale_tril = torch.diag(self._std)
                batch_dim = batch_mean.shape[0]
                batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
                action_distribution = distributions.MultivariateNormal(
                    batch_mean,
                    scale_tril=batch_scale_tril,
                )
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################


class ConcatMLP(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self._dim)
        return super().forward(flat_inputs, **kwargs)

class MLPPolicyDeterministic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, *args, **kwargs):
        kwargs['deterministic'] = True
        super().__init__(*args, **kwargs)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        return {"Training Loss": loss.item()}
    
class MLPPolicyStochastic(MLPPolicy):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, entropy_coeff, *args, **kwargs):
        kwargs['deterministic'] = False
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: sample actions from the gaussian distribrution given by MLPPolicy policy when providing the observations.
        # Hint: make sure to use the reparameterization trick to sample from the distribution
        
        return ptu.to_numpy(action)
        
    def update(self, observations, q_fun):
        # TODO: update the policy and return the loss
        ## Hint you will need to use the q_fun for the loss
        ## Hint: do not update the parameters for q_fun in the loss
        ## Hint: you will have to add the entropy term to the loss using self.entropy_coeff
        return {"Loss": loss.item()}
    
#####################################################