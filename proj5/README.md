# Project 5: 3D Pose Estimation and objectron for object selection

# Setup
- Install <a href="https://conda.io/miniconda.html">Miniconda</a>. It doesn't matter whether you use 2.7 or 3.6 because we will create our own environment anyways.
- Create a conda environment, using the appropriate command. On Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac'): `conda env create -f proj5_env_<OS>.yml`
- This should create an environment named `proj5`. Activate it using the following Windows command: `activate proj5` or the following MacOS / Linux command: `source activate proj5`.
- Install tensorflow via `pip install tensorflow`
- Install Mediapipe via `pip install mediapipe`
- Special notice to Windows user, it seems mediapipe does not have good support for Windows system, and so you are strongly suggested to use other operating systems or try to set up a CoLab environment. There may be some path issues that you have to modify if you want to run it on Colab.
- Special notice to Mac user, it seems many students also have problem with configuring their running environment even though mediapipe claims it can be installed via pip. So if you meet installation problems, please also consider using Colab, or try to follow installation instructions posed by Mediapipe's official website.
- Run `pip install -e .` (Maybe not work for Mac user)
- Run the notebook using: `jupyter notebook proj5_code/proj5.ipynb`
- After finishing all your functions, please check their correctness via `pytest` in the folder `proj5_unit_tests`
- Generate the submission once you're finished using `python zip_submission.py`
