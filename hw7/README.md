# Running this homework

First off, we need a different version of python for this homework.
Create a new venv using `python3.9 -m venv venv`. 
Then, install the requirements: run `pip3 install -r requirements.txt` in the main folder of this repository.

You can run `./roble/ppo.py` and `./roble/sac.py` by using the `../run_hw7_sim2real.py`. 
This loads the config file found in `../conf/config_hw7.yaml`.
Both PPO and SAC use the same config file. In this file, `../conf/config_hw7.yaml`, you can see their respective `ppo:` and `sac:` sections.
Like the other homeworks, we use hydra to manage config files. This means that you can override the configs like so:

```bash
python run_hw7_sim2real.py meta.sac_instead=False meta.add_to_runname=Q2.1    # Part 2.1
python run_hw7_sim2real.py meta.sac_instead=True meta.add_to_runname=Q2.2     # Part 2.2
python run_hw7_sim2real.py sim2real.history_len=0 meta.add_to_runname=Q3h0    # Part 3
python run_hw7_sim2real.py sim2real.history_len=4 meta.add_to_runname=Q3h4    # Part 3
python run_hw7_sim2real.py sim2real.gaussian_obs_scale=0.01 sim2real.gaussian_act_scale=0.01 meta.add_to_runname=Q3g0.0 # Part 3
python run_hw7_sim2real.py sim2real.gaussian_obs_scale=0.1 sim2real.gaussian_act_scale=0.1 meta.add_to_runname=Q3g0.1   # Part 3
python run_hw7_sim2real.py sim2real.gaussian_obs_scale=1.0 sim2real.gaussian_act_scale=1.0 meta.add_to_runname=Q3g1     # Part 3
```

An attentive reader might notice that each of these run lines correspond to the different commands we ask you to run in the PDF...

If you get errors that complain about puppersim not being installed, or some other similar issue, you will need to fix your own pythonpath.
If you need help, contact @velythyl in the discord channel.

# IMPLEMENTATON

Fill in the wrappers found in the `sim2real` folder.

# PPO

Run PPO with the default config.

# SAC

Run SAC with the default config.

# Sim2Real ablation

Run the different sim2real wrapper parameter values we ask of you (history=0, history=4, gaussian noise=0.01, gaussian noise=0.1, gaussian noise=1.0).

# FOR MORE DETAILS

Read through the PDF!