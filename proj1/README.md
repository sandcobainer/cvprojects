# CS 6476 - Fall 20 - Project 1: [Image Filtering and Hybrid Images](https://dellaert.github.io/20F-6476/proj1/)

# Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj1_env_<OS>.yml`
3. This should create an environment named 'proj1'. Activate it using the Windows command, `activate proj1` or the MacOS / Linux command, `conda activate proj1`
4. Install the project package, by running `pip install -e .` inside the repo folder. This should be unnecessary for Project1, but is good practice when setting up a new `conda` environment that may have `pip` requirements.
5. Run the notebook using `jupyter notebook ./proj1_code/proj1.ipynb`
6. Ensure that all sanity checks are passing by running `pytest proj1_unit_tests` inside the `proj1_code` folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`
