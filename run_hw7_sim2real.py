import os
import time

import comet_ml
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hw7.roble import ppo, sac

PATH = "./hw7/runs"

class Log:
    def __init__(self, cfg, PATH):
        from hw7.roble.utils.logging_utils import Logger as TableLogger
        self._logger = TableLogger()
        self._logger.add_folder_output(folder_name=f"{PATH}")
        self._logger.add_tabular_output(file_name=f"{PATH}/log_data.csv")
        os.makedirs(PATH, exist_ok=True)
        with open(f"{PATH}/conf.yaml", "w") as fd:
            fd.write(OmegaConf.to_yaml(cfg))
            fd.flush()

    def log_dict(self, dico):
        for k, v in dico.items():
            if isinstance(v, list) and len(v) == 0:
                continue
            self._logger.record_tabular_misc_stat(k, v)
        self._logger.dump_tabular()


@hydra.main(config_path="conf", config_name="config_hw7")
def my_main(args: DictConfig):
    global PATH


    os.chdir(get_original_cwd())

    run_name = f"{args.meta.env_id}__{args.meta.exp_name}__{args.meta.add_to_runname}__{args.meta.seed}__{int(time.time())}"
    args.meta.run_name = run_name
    PATH = f"{PATH}/{args.meta.run_name}"


    logger = Log(args, PATH)
    if args.meta.track:
        experiment = comet_ml.Experiment(
        api_key=your key,
        project_name=project name
        )
        
        experiment.add_tag("hw3")
        experiment.set_name(args.meta.run_name)
        experiment.set_filename(fname="cometML_test")
        logger._logger.set_comet_logger(experiment)

    def get_arg_dict(args):
        dico = dict(vars(args))
        return dico["_content"]

    def flatten_conf(conf1, conf2):
        dico = get_arg_dict(conf1)
        dico.update(get_arg_dict(conf2))

        args = OmegaConf.create(dico)
        return args

    sim2real = args.sim2real
    new_args = flatten_conf(args.meta, OmegaConf.create({"sim2real": get_arg_dict(sim2real)}))
    if args.meta.sac_instead:
        args = flatten_conf(new_args, args.sac)

        args.buffer_size = int(args.buffer_size)

        sac.train(args, logger, PATH)
    else:
        args = flatten_conf(new_args, args.ppo)

        args.target_kl = None if args.target_kl == "None" else args.target_kl

        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        ppo.train(args, logger, PATH)



if __name__ == "__main__":
    my_main()
