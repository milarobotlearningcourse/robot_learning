import os
import time

import sys
print(sys.path)


from hw2.roble.agents.mb_agent import MBAgent
from hw2.roble.infrastructure.rl_trainer import RL_Trainer
import hydra, json
from omegaconf import DictConfig, OmegaConf

class MB_Trainer(object):

    def __init__(self, params):

        self._params = params
        ################
        ## RL TRAINER
        ################

        self._rl_trainer = RL_Trainer(self._params , agent_class =  MBAgent)

    def run_training_loop(self):

        self._rl_trainer.run_training_loop(
            self._params['alg']['n_iter'],
            collect_policy = self._rl_trainer._agent._actor,
            eval_policy = self._rl_trainer._agent._actor,
            )

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))
    import os
    print("Command Dir:", os.getcwd())
    # print ("params: ", json.dumps(params, indent=4))
    if cfg['env']['env_name']=='reacher-roble-v0':
        cfg['env']['max_episode_length']=200
    if cfg['env']['env_name']=='cheetah-roble-v0':
        cfg['env']['max_episode_length']=500
    if cfg['env']['env_name']=='obstacles-roble-v0':
        cfg['env']['max_episode_length']=100
    params = vars(cfg)
    print ("params: ", params)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################


    logdir_prefix = 'hw2_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    exp_name = logdir_prefix + cfg.env.exp_name + '_' + cfg.env.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logging.logdir = logdir
        cfg.logging.exp_name = exp_name

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = MB_Trainer(cfg)
    trainer.run_training_loop()

