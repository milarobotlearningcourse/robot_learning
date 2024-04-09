
# Installation

These installation instructions are created for Ubuntu 20.04. If you are using a different OS you made need to make some changes to the installing instructions. 


## Ubuntu 20.04

```
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf
```

## Install Python Environment


There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3:
	```
	conda create -n roble python=3.8
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate roble
	```

	4. Install the requirements into this conda environment
	```
	pip install --user -r requirements.txt
	```

This conda environment requires activating every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:
	```
  pip install -r requirements.txt
  cd <path_to_hw1>
  pip install -e .
	```


## Debugging issues with installing `mujoco-py`

If you run into issues with installing `mujoco-py` (especially on MacOS), here are a few common pointers to help:
  1. If you run into GCC issues, consider switching to GCC7 (`brew install gcc@7`)
  2. [Try this](https://github.com/hashicorp/terraform/issues/23033#issuecomment-543507812) if you run into developer verification issues (use due diligence when granting permissions to code from unfamiliar sources)
  3. StackOverflow is your friend, feel free to shamelessly look up your error and get in touch with your classmates or instructors
  4. If nothing works and you are frustrated beyond repair, consider using the Docker or Colab version of the homework!
