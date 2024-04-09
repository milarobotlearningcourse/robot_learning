Assignment 1: Imitation Learning\

The goal of this assignment is to experiment with imitation learning,
including direct behavior cloning, inverse dynamics models (IDM) and the
DAgger algorithm. In lieu of a human demonstrator, demonstrations will
be provided via an expert policy that we have trained for you. Your
goals will be to set up behavior cloning, training an IDM to leverage
unlabelled data and DAgger, and compare their performance on a few
different continuous control tasks from the OpenAI Gym benchmark suite.
Turn in your report and code as described in .

The starter-code for this assignment can be found at

<https://github.com/milarobotlearningcourse/robot_learning>\

You have the option of running the code either on Google Colab (not
completely supported) or on your own machine. Please refer to the
`README` for more information on setup.

Behavioral Cloning
==================

1.  The starter code provides an expert policy for each of the MuJoCo
    tasks in OpenAI Gym and 2 expert demonstrations containing
    successful rollouts with (state, action, next\_state) data.
    Additionally, the code provides an **unlabelled** dataset without
    actions of successful rollouts with (state, next\_state) data. In
    Section [1.1](#sec:idm){reference-type="ref" reference="sec:idm"} we
    will learn how to leverage the unlabelled data to improve the
    performance of the BC agent.

    Fill in the blanks in the code marked with `TODO` to implement
    behavioral cloning. A command for running behavioral cloning is
    given in the Readme file.

    We recommend that you read the files in the following order. For
    some files, you will need to fill in blanks, labeled `TODO`.

    -   `run_hw1.py`

    -   `infrastructure/rl_trainer.py`

    -   `agents/bc_agent.py`

    -   `policies/MLP_policy.py`

    -   `infrastructure/replay_buffer.py`

    -   `infrastructure/utils.py`

    -   `infrastructure/pytorch_utils.py`

2.  Run behavioral cloning (BC) and report results on two tasks: the Ant
    environment, where a behavioral cloning agent should achieve at
    least 30% of the performance of the expert, and one environment of
    your choosing where it does not. Here is how you can run the Ant
    task:

    ``` {.bash language="bash"}
    python run_hw1_bc.py 
    ```

    This code uses [hydra](https://hydra.cc/docs/intro/) to organize
    experimental runs and data. There is a *YAML* file in
    *conf/config\_hw1.yaml* that is used to control the parameters
    passed to the code for training. The YAML file looks like

    ``` {escapechar="@"}
    env: 
      expert_policy_file: ../../../hw1/roble/policies/experts/Ant.pkl
      expert_data: ../../../hw1/roble/expert_data/labelled_data_Ant-v2.pkl
      expert_unlabelled_data: ../../../hw1/roble/expert_data/unlabelled_data_Ant-v2.pkl
      exp_name: "bob"
      env_name: Ant-v2 # choices are [Ant-v2, Humanoid-v2, Walker2d-v2, HalfCheetah-v2, Hopper-v2]
      max_episode_length: 100 
      render: false
      
    alg:
      num_rollouts: 5
      train_idm: false
      do_dagger: false
      idm_batch_size: 128
      idm_num_epochs: 100
      .
      .
      .
      
    ```

    When providing results, report the mean and standard deviation of
    your policy's **average reward** over multiple trajectories in a
    table, and state which task was used. When comparing one that is
    working versus one that is not working, be sure to set up a fair
    comparison in terms of network size, amount of data, and number of
    training iterations. Provide these details (and any others you feel
    are appropriate) in the table caption. Submit your log file
    *data/\.../log\_file.log* on Gradescope as *ant1-2.log* for your Ant
    run and *custom1-2.log* for the run of your choosing.

    **Note**: What "report the mean and standard deviation"means is that
    your `eval_batch_size` should be greater than `ep_len`, such that
    you're collecting multiple rollouts when evaluating the performance
    of your trained policy. For example, if `ep_len` is 1000 and
    `eval_batch_size` is 5000, then you'll be collecting approximately 5
    trajectories (maybe more if any of them terminate early), and the
    logged `eval _reward_Average` and `eval_reward_Std` represent the
    mean/std of your policy over these 5 rollouts. Make sure you include
    these parameters in the table caption as well.

    **Note**: Make sure to set both flags
    [`video_log_freq`]{style="color: red"} in *conf/config\_hw1.yaml* to
    be greater than 1 and [`render`]{style="color: red"} to true when
    training your **final** agent for videos recording as requested in
    section [3](#sec:turn-it-in){reference-type="ref"
    reference="sec:turn-it-in"}.

3.  Experiment with one set of hyperparameters that affects the
    performance of the behavioral cloning agent, such as the amount of
    training steps, the amount of expert data provided, or something
    that you come up with yourself. For one of the tasks used in the
    previous question, show a graph of how the BC agent's performance
    varies with the value of this hyperparameter. In the caption for the
    graph, state the hyperparameter and a brief rationale for why you
    chose it.

Inverse Dynamics {#sec:idm}
----------------

An Inverse Dynamics Model (IDM) learns what actions are taken between
consecutive pairs of states. These models are very sample-efficient,
meaning that they become accurate predictors even with very little
expert data. In this section, we will train an IDM on the expert data
provided, and use it to predict the actions of the unlabelled dataset.
In this way, we will be able to leverage unlabelled data to augment the
dataset in which we train the BC agent on. This pipeline was
successfully implemented in [this recent paper by
OpenAI](https://openai.com/research/vpt).

Step-by-step guide:

-   Unzip the unlabelled data in
    *robot\_learning/hw1/roble/expert\_data/unlabelled.zip* and make
    sure the extracted data folder is moved inside
    *robot\_learning/hw1/roble/expert\_data/* again.

-   Implement the IDM logic in *hw1/roble/agents/bc\_agent.py* and
    *hw1/roble/policies/MLP\_policy.py*

-   Complete the TODO in *hw1/roble/infrastructure/rl\_trainer.py* to
    plot the learning curve of the IDM and add it to your report.

-   Set *train\_idm* to True in the config file and make sure the
    *expert\_unlabelled\_data* variable in the config file also points
    to the correct path. E.g. :
    *../../../hw1/roble/expert\_data/unlabelled/unlabelled\_data\_Ant-v2.pkl*

Once the IDM is correctly implemented, run the code again with the
config variable *train\_idm* set to True:

``` {.bash language="bash"}
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.train_idm=true 
```

This will create another dataset named *labelled\_data\_envname.pkl*
that should automatically be placed at the expert data directory.
Furthermore, with no interruptions in the execution, your BC agent will
be trained on the labelled dataset.

For this Section of the assignment, add the plot of the training curve
that you implemented in the *train\_idm()* (see the TODO in the
*run\_training\_loop()* function in
*hw1/roble/infrastructure/rl\_trainer.py*) function, and make a bar plot
that compares the performance that the BC agent achieves when trained on
1) the original expert data and 2) the labelled data. Use the Ant-v2 and
HalfCheetah environments. You can modify some hyperparameters but make
sure to set up a fair comparison between the 2 settings.

DAgger
======

1.  Once you've filled in all of the `TODO` commands, you should be able
    to run DAgger by editing the *config.yaml* file here

    ``` {.bash language="bash"}
    alg:
      num_rollouts: 5
      do_dagger: true
    ```

    and running again

    ``` {.bash language="bash"}
      python run_hw1_bc.py alg.n_iter=5 alg.do_dagger=true alg.train_idm=false
    ```

2.  Run DAgger and report results on the two tasks you tested previously
    with behavioral cloning (i.e., Ant + another environment). Report
    your results in the form of a learning curve, plotting the number of
    DAgger iterations vs. the policy's mean return, with error bars to
    show the standard deviation. Include the performance of the expert
    policy and the behavioral cloning agent on the same plot (as
    horizontal lines that go across the plot). In the caption, state
    which task you used, and any details regarding network architecture,
    amount of data, etc. (as in the previous section). Submit the log
    file of your Ant run on Gradescope as *dagger\_ant2-2.log*.

Turning it in {#sec:turn-it-in}
=============

1.  **Submitting the PDF.  ** Make a PDF report containing: Table 1 for
    a table of results from Question 1.2, Figure 1 for Question 1.3. and
    Figure 2 with results from question 2.2.

    You do not need to write anything else in the report, just include
    the figures with captions as described in each question above. Full
    points for clear captions. See the handout at

    <http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf>\

    for notes on how to generate plots.

2.  **Submitting log files on the autograder.** Make sure to submit all
    the log files that are requested on Gradescope you can find them in
    your log directory */data* by default.

3.  **Submitting the code, experiment runs, and GIFs.  ** In order to
    turn in your code and experiment logs, create a folder that contains
    the following:

    -   A folder named `run_logs` with experiments for both the
        behavioral cloning (part 2, not part 3) exercise and the DAgger
        exercise. Note that you can include multiple runs per exercise
        if you'd like, but you must include at least one run (of any
        task/environment) per exercise. These folders can be copied
        directly from the `output` folder into this new folder. You must
        also provide a GIF of your final policy for each question. To
        enable video logging, set both flags `video_log_freq` to greater
        than 1 and `render` to be true in *conf/config\_hw1.yaml* and
        run your experiment. The GIFs will be located in the
        *outputs/path\_to\_experiment/videos/eval\_step\_x.gif* path.

    -   The `ift6163` folder with all the `.py` files, with the same
        names and directory structure as the original homework
        repository. Also include the commands (with clear
        hyperparameters) and *conf/config\_hw1.yaml* file that we need
        in order to run the code and produce the numbers that are in
        your figures/tables (e.g. run "python run\_hw1.py --ep\_len 200"
        to generate the numbers for Section 2 Question 2) in the form of
        a README file.

    As an example, the unzipped version of your submission should result
    in the following file structure. **Make sure to include the prefix
    `q1_` and `q2_`.**

    for tree= font=, grow'=0, child anchor=west, parent anchor=south,
    anchor=west, calign=first, edge path= (!u.south west) +(7.5pt,0) \|-
    node\[fill,inner sep=1.25pt\] (.child anchor); , before typesetting
    nodes= if n=1 insert before=\[,phantom\] , fit=band, before
    computing xy=l=15pt, \[submit.zip \[hw1 \[run\_logs \[q1\_bc\_ant
    \[log\_data.csv\] \[videos\] \] \[q2\_dagger\_ant \[log\_data.csv\]
    \[videos\] \] \[\...\] \] \[roble \[agents \[bc\_agent.py\] \[\...\]
    \] \[policies \[\...\] \] \[\...\] \] \[\...\] \] \[conf
    \[config\_hw1.yaml\] \] \[Dockerfile\] \[diff.txt\] \[\...\] \]

You also need to include a diff of your code compared to the starter
homework code. You can use the command

``` {.bash language="bash"}
git diff 8ea2347b8c6d8f2545e06cef2835ebfd67fdd608 >> diff.txt
```

If you are a Mac user, **do not use the default "Compress" option to
create the zip**. It creates artifacts that the autograder does not
like. You may use `zip -vr submit.zip submit -x "*.DS_Store"` from your
terminal.

Turn in your assignment on Gradescope. Upload the zip file with your
code and log files to **HW1 Code**, and upload the PDF of your report to
**HW1**.
