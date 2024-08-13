# Tactix Game Project

This project is made for my bachelor thesis "Development and Evaluation of Deep Q-Networks for the Game TacTix on Different Board Sizes" at Technical University of Berlin / DAI-Labor. This project implements the Tactix game along with AI agents using MCTS and DQN algorithms.

## Setup

To set up the project, run the following commands:

```sh
conda env create -f environment.yml
conda activate tactix-game-env


## if you are going to add dependencies to setup.py in order to distrubute your project you should also update the install_requires section in setup.py 

## In order to update the environment when you add a new dependency :

add the requirement to requirement.txt and then run the command:

conda env update -f environment.yml

## To activate and deactivate the environment

conda activate tactix-game-env

conda deactivate


## Testing

## to run tests run this command from the root directory:

python -m unittest tests/test_logic.py ## the name can be specified

## or you can run all the tests :

python -m unittest discover -s tests

