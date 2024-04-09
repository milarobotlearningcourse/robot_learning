**Homework 3 : Q-Learning Algorithms**\

> Part 1 of this assignment requires you to implement and evaluate
> Q-learning for playing Atari games. The Q-learning algorithm was
> covered in lecture, and you will be provided with starter code. This
> assignment will run faster on a GPU, though it is possible to complete
> on a CPU as well. Note that we use convolutional neural network
> architectures in this assignment. Please start early! For references
> to this type of approach, see this
> [paper](https://arxiv.org/abs/1312.5602) and this
> [paper](https://arxiv.org/abs/1509.02971).

Part 1: DQN
===========

We will be building on the code that we have implemented in the first
two assignments. All files needed to run your code are in the `hw3`
folder. Files to edit:

-   `infrastructure/rl_trainer.py`

-   `infrastructure/utils.py`

-   `policies/MLP_policy.py`

In order to implement deep Q-learning, you will be writing new code in
the following files:

-   `agents/dqn_agent.py`

-   `critics/dqn_critic.py`

-   `policies/argmax_policy.py`

There are two new package requirements (`opencv-python` and
`gym[atari]`) beyond what was used in the first two assignments; make
sure to install these with `pip install -r requirements.txt` if you are
running the assignment locally.

Implementation 
--------------

The first phase of the assignment is to implement a working version of
Q-learning. The default code will run the `Ms. Pac-Man` game with
reasonable hyperparameter settings. Look for the `# TODO` markers in the
files listed above for detailed implementation instructions. You may
want to look inside `infrastructure/dqn_utils.py` to understand how the
(memory-optimized) replay buffer works, but you will not need to modify
it.

Once you implement Q-learning, answering some of the questions may
require changing hyperparameters, neural network architectures, and the
game, which should be done by changing the command line arguments passed
to `run_hw3_dqn.py` or by modifying the parameters of the `Args` class
from within the Colab notebook.

To determine if your implementation of Q-learning is correct, you should
run it with the default hyperparameters on the `Ms. Pac-Man` game for 1
million steps using the command below. Our reference solution gets a
return of 1500 in this timeframe. On Colab, this will take roughly 3 GPU
hours. If it takes much longer than that, there may be a bug in your
implementation.

To accelerate debugging, you may also test on `LunarLander-v3`, which
trains your agent to play Lunar Lander, a 1979 arcade game (also made by
Atari) that has been implemented in OpenAI Gym. Our reference solution
with the default hyperparameters achieves around 150 reward after 350k
timesteps, but there is considerable variation between runs, and without
the double-Q trick, the average return often decreases after reaching
150. We recommend using `LunarLander-v3` to check the correctness of
your code before running longer experiments with `MsPacman-v0`.

Evaluation
----------

Once you have a working implementation of Q-learning, you should prepare
a report. The report should consist of one figure for each question
below. You should turn in the report as one PDF and a zip file with your
code. If your code requires special instructions or dependencies to run,
please include these in a file called `README` inside the zip file.
Also, provide the log file of your run on gradescope named as
`pacman_1.csv`.

#### Question 1: basic Q-learning performance (DQN).

Include a learning curve plot showing the performance of your
implementation on `Ms. Pac-Man`. The x-axis should correspond to a
number of time steps (consider using scientific notation), and the
y-axis should show the average per-epoch reward as well as the best mean
reward so far. These quantities are already computed and printed in the
starter code. They are also logged to the `data`. Be sure to label the
y-axis, since we need to verify that your implementation achieves
similar reward as ours. You should not need to modify the default
hyperparameters in order to obtain good performance, but if you modify
any of the parameters, list them in the caption of the figure. The final
results should use the following experiment name:

``` {.bash language="bash"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=MsPacman-v0 env.exp_name=q1
```

#### Question 2: double Q-learning (DDQN).

Use the double estimator to improve the accuracy of your learned Q
values. This amounts to using the online Q network (instead of the
target Q network) to select the best action when computing target
values. Compare the performance of DDQN to vanilla DQN. Since there is
considerable variance between runs, you must run at least three random
seeds for both DQN and DDQN. You may use `LunarLander-v3` for this
question. The final results should use the following experiment names:

``` {.bash language="bash" breaklines="true"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_1 logging.seed=1
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_2 logging.seed=2
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_3 logging.seed=3
```

``` {.bash language="bash" breaklines="true"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_1 alg.double_q=true logging.seed=1
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_2 alg.double_q=true logging.seed=2
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_3 alg.double_q=true logging.seed=3
```

Submit the run logs for all the experiments above. In your report, make
a single graph that averages the performance across three runs for both
DQN and double DQN. See `scripts/read_results.py` for an example of how
to read the evaluation returns from Tensorboard logs.

#### Question 3: experimenting with hyperparameters.

Now, let's analyze the sensitivity of Q-learning to hyperparameters.
Choose one hyperparameter of your choice and run at least three other
settings of this hyperparameter in addition to the one used in Question
1, and plot all four values on the same graph. Your choice is what you
experiment with, but you should explain why you chose this
hyperparameter in the caption. Examples include (1) learning rates; (2)
neural network architecture for the Q network, e.g., number of layers,
hidden layer size, etc; (3) exploration schedule or exploration rule
(e.g. you may implement an alternative to $\epsilon$-greedy and set
different values of hyperparameters), etc. Discuss the effect of this
hyperparameter on performance in the caption. You should find a
hyperparameter that makes a nontrivial difference in performance. Note:
you might consider performing a hyperparameter sweep to get good results
in Question 1, in which case it's fine to just include the results of
this sweep for Question 3 as well while plotting only the best
hyperparameter setting in Question 1. The final results should use the
following experiment name:

``` {.bash language="bash" breaklines="true"}
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam1
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam2
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam3
```

You can replace `LunarLander-v3` with `PongNoFrameskip-v4` or
`MsPacman-v0` if you would like to test on a different environment.

Part 2: DDPG
============

Implement the DDPG algorithm.

In order to implement Deep Deterministic Policy Gradient (DDPG), you
will be writing new code in the following files:

-   `agents/ddpg_agent.py`

-   `critics/ddpg_critic.py`

-   `policies/MPL_policy.py`

DDPG is programmed a little differently than the RL algorithms so far.
DDPG does not use n-step returns to estimate the advantage given a large
batch of on-policy data. Instead, DDPG is off-policy. DDPG trains a
Q-Function $Q(\textbf{s}_t, \textbf{a}_t, \phi)$ to estimate the policy
*reward-to-go* if for a state and action. This model can then be used as
the objective to optimize the current policy.

$$\begin{split}
  \nabla_{\theta^\mu} J &\approx
  \mathbb{E}_{\textbf{s}_t\sim \rho^\beta}\left[\nabla_{\theta}
    Q(\textbf{s}_t, \textbf{a}_t| \phi)|_{s = \textbf{s}_t, a = \mu(s_t | \theta)}
                   \right] \\
                 & =
    \mathbb{E}_{s_t \sim \rho^\beta}\left[\nabla_{a} Q(\textbf{s}_t, \textbf{a}_t| \phi)|_{s = \textbf{s}_t, a = \mu(\textbf{s}_t)}
    \nabla_{\theta_\mu} \mu(s | \theta)|_{s = s_t} \right]
  \end{split}$$

See the lecture slides on ddpg for how to implement the code.

#### Question 4: Experiments (DDPG)

For this question, the goal is to implement DDPG and tune a few of the
hyperparameters to improve the performance. Try different update
frequencies for the Q-Function. Also, try different learning rates for
the Q-function and actor. First, try different learning rates.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py env.exp_name=q4_ddpg_up<b>_lr<r> alg.rl_alg=ddpg
env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=<b> alg.learning_rate=<lr>

python run_hw3_ql.py  env.exp_name=q4_ddpg_up<b>_lr<r>  alg.rl_alg=ddpg
env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=<b> alg.learning_rate=<lr>

python run_hw3_ql.py   env.exp_name=q4_ddpg_up<b>_lr<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_critic_updates_per_agent_update=<b> alg.learning_rate=<lr>
```

Next, try different update frequencies for training the policies.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py    env.exp_name=q4_ddpg_up<b>_lr<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b> alg.learning_rate=<lr>

python run_hw3_ql.py  env.exp_name=q4_ddpg_up<b>_lr<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b> alg.learning_rate=<lr>

python run_hw3_ql.py    env.exp_name=q4_ddpg_up<b>_lr<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b> alg.learning_rate=<lr>
```

Submit the learning graphs from these experiments along with the
write-up for the assignment.

#### Question 5: Best parameters on a more difficult task

After you have completed the parameter tuning on the simpler
*InvertedPendulum-v2* environment, use those parameters to train a model
on the more difficult *HalfCheetah-v2* environment.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw4.py   env.exp_name=q5_ddpg_hard_up<b>_lr<r>  alg.rl_alg=ddpg
env.env_name=HalfCheetah-v2 env.atari=false
```

Include the learning graph from this experiment in the write-up as well.
Also, provide the log file of your run on gradescope named as
`half_cheetah_5.csv`.

Part 3: TD3
===========

In order to implement Twin Delayed Deep Deterministic Policy Gradient
(TD3), you will be writing new code in the following files:

-   `critics/td3_critic.py`

This is a relatively small change to DDPG to get TD3. Implement the
additional target q function for
[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html)

See the lecture slides on td3 for how to implement the code.

#### Question 6: TD3 tuning

Again, the hyperparameters for this new algorithm need to be tuned as
well using *InvertedPendulum-v2*. Try different values for the noise
being added to the target policy $\rho$ when computing the target
values. Also try different Q-Function network structures. Start with
trying different values for $\rho$.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py  env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.td3_target_policy_noise=0.05

python run_hw3_ql.py  env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.td3_target_policy_noise=0.1

python run_hw3_ql.py  env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=td3 env.env_name=InvertedPendulum-v2 env.atari=false alg.td3_target_policy_noise=0.2
```

Next, try different update frequencies for training the policies.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b>

python run_hw3_ql.py env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b>

python run_hw3_ql.py env.exp_name=q6_td3_shape<s>_rho<r>  alg.rl_alg=ddpg env.env_name=InvertedPendulum-v2 env.atari=false alg.num_agent_train_steps_per_iter=<b>
```

Include the results from these hyper-parameter experiments in the
assignment write-up. Make sure to comment clearly on the parameters you
studied and why which settings have worked better than others.

#### Question 7: Evaluate TD3 compared to DDPG

In this last question, evaluate TD3 compared to DDPG. Using the best
parameter setting from **Q6** train TD3 on the more difficult
environment used for **Q5**.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py env.exp_name=q6_td3_shape<s>_rho<r> alg.rl_alg=td3 env.env_name=HalfCheetah-v2 env.atari=false
```

Include the learning graph from this experiment in the write-up for the
assignment. Make sure to comment on the different performance between
DDPG and TD3 and what makes the performance different. Also, provide the
log file of your run on gradescope named as `half_cheetah_7.csv`.

#### Bonus: For finding issues or adding features

Keeping up with the latest research often means adding new types of
programming questions to the assignments. If you find issues with the
code and provide a solution, you can receive bonus points. Also, adding
features that help the class can also result in bonus points for the
assignment.

Part 4: SAC 
===========

In order to implement Soft Actor-Critic (SAC), you will be writing new
code in the following files:

-   `agents/sac_agent.py`

-   `critics/sac_critic.py`

-   MLPPolicyStochastic in `policies/MLP_policy.py`

SAC is similar to TD3 except for its central feature, which is entropy
regularization. The policy is trained to maximize a trade-off between
expected return and entropy, a measure of randomness in the policy. The
algorithm and some relevant resources are available
[here](https://spinningup.openai.com/en/latest/algorithms/sac.html).
**Bonus**: Add linear annealing to alpha (the entropy coefficient).

#### Question 8: SAC entropy tuning

Again, the hyperparameters for this new algorithm need to be tuned as
well using *InvertedPendulum-v2*. Try different values for the entropy
coefficient $\alpha$ being added to the loss.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py    env.exp_name=q8_sac_alpha<a>  alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.entropy_coeff=0.05

python run_hw3_ql.py    env.exp_name=q8_sac_alpha<a>  alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.entropy_coeff=0.05

python run_hw3_ql.py    env.exp_name=q8_sac_alpha<a>  alg.rl_alg=sac env.env_name=InvertedPendulum-v2 env.atari=false alg.entropy_coeff=0.05
```

#### Question 9: Evaluate SAC compared to TD3

In this last question, evaluate SAC compared to TD3. Using the best
parameter setting from **Q6** train SAC on the more difficult
environment used for **Q5**.

``` {.bash escapechar="@" language="bash" breaklines="true"}
python run_hw3_ql.py    env.exp_name=q9_td3_shape<s>_rho<r> alg.rl_alg=td3 env.env_name=HalfCheetah-v2 env.atari=false
```

Include the learning graph from this experiment in the write-up for the
assignment. Make sure to comment on the different performance between
TD3 and SAC and what makes the performance different. Also, provide the
log file of your run on gradescope named as `half_cheetah_9.csv`.

Submission
==========

We ask you to submit the following content on the course GradeScope :

Submitting the PDF.
-------------------

Your report should be a PDF document containing the plots and responses
indicated in the questions above.

Submitting log files on the autograder. 
---------------------------------------

Make sure to submit all the log files that are requested by GradeScope
AutoGrader, you can find them in your log directory `/data/exp_name/` by
default.

Submitting code, experiment runs, and videos. 
----------------------------------------------

In order to turn in your code and experiment logs, create a folder that
contains the following:

-   A folder named `data` with ONLY the experiment runs from this
    assignment. **Do not change the names originally assigned to the
    folders, as specified by `exp_name` in the instructions.**

-   The `roble` folder with all the `.py` files, with the same names and
    directory structure as the original homework repository (not
    including the `outputs/` folder). Additionally, include the commands
    (with clear hyperparameters) and the config file
    `conf/config_hw3.yaml` file that we need in order to run the code
    and produce the numbers that are in your figures/tables (e.g. run
    "python run\_hw3.py env.ep\_len=200") in the form of a README file.
    Finally, your plotting script should also be submitted, which should
    be a Python script (or jupyter notebook) such that running it can
    generate all plots from your pdf. This plotting script should
    extract its values directly from the experiments in your `outputs/`
    folder and **should not have hardcoded reward values**.

-   You must also provide a video of your final policy for each question
    above. To enable video logging, set both flags video *log\_freq* to
    be greater than 0 and render to be true in `conf/config_hw3.yaml`
    **before** running your experiments. Videos could be fin as `.mp4`
    files in the folder: `data/exp_name/videos/`. **Note :** For this
    homework, the atari envs should be in the folder `gym` and the other
    in the folder `video`.

As an example, the unzipped version of your submission should result in
the following file structure. **Make sure to include the prefix
`q1_`,`q2_`,`q3_`,`q4_`, and `q5_`.**

You also need to include a diff in your code compared to the starter
homework code. You can use the command.

``` {.bash language="bash"}
git diff 8ea2347b8c6d8f2545e06cef2835ebfd67fdd608 >> diff.txt
```

1.  If you are a Mac user, **do not use the default "Compress" option to
    create the zip**. It creates artifacts that the autograder does not
    like. You may use `zip -vr submit.zip submit -x "*.DS_Store"` from
    your terminal.

2.  Turn in your assignment on Gradescope. Upload the zip file with your
    code and log files to **HW3 Code**, and upload the PDF of your
    report to **HW3**.
