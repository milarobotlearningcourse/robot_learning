import os
import numpy as np
import time
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader


from hw1.roble.infrastructure.rl_trainer import RL_Trainer
# from hw1.roble.infrastructure.idm import IDM
from hw1.roble.agents.bc_agent import BCAgent
from hw1.roble.policies.loaded_gaussian_policy import LoadedGaussianPolicy

class BC_Trainer(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, params):

        ################
        ## RL TRAINER
        ################

        self._rl_trainer = RL_Trainer(self._params, agent_class=BCAgent) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################
        ### Correcting for hydra logging folder. 
        print('Loading expert policy from...',  self._params["env"]['expert_policy_file'])
        self._loaded_expert_policy = LoadedGaussianPolicy( self._params["env"]['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):
        results = self._rl_trainer.run_training_loop(
            n_iter=self._params['alg']['n_iter'],
            initial_expertdata=self._params['env']['expert_data'],
            collect_policy=self._rl_trainer._agent._actor,
            eval_policy=self._rl_trainer._agent._actor,
            relabel_with_expert=self._params['alg']['do_dagger'],
            expert_policy=self._loaded_expert_policy,
        )

        return results

    
import hydra, json
from omegaconf import DictConfig, OmegaConf

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    params = vars(cfg)
    # print ("params: ", json.dumps(params, indent=4))
    # print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if cfg.alg.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert cfg.alg.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        if cfg.alg.train_idm:
            logdir_prefix = 'q1_idm_'
        else:
            # Use this prefix when submitting. The auto-grader uses this prefix.
            logdir_prefix = 'q1_'
        assert cfg.alg.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################
    trainer = BC_Trainer(cfg)
    
    
    out = trainer.run_training_loop()
    return out
