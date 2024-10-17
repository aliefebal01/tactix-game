# Tactix Game Project

This project is developed as part of my bachelor's thesis, "Development and Evaluation of Deep Q-Networks for the Game TacTix on Different Board Sizes," at the Technical University of Berlin / DAI-Labor. The project includes the implementation of the Tactix game along with AI agents using MCTS (Monte Carlo Tree Search) and DQN (Deep Q-Network) algorithms.

# Setup

### Environment Setup

To set up the required environment, run the following commands:

 Create the environment using the environment.yml file
```bash
conda env create -f environment.yml
```

Activate the environment
```bash
conda activate tactix-game-env
```

### Adding Dependencies

If you need to add dependencies to setup.py for distributing your project, ensure you update the install_requires section in setup.py. For additional packages, follow these steps:

1. Add the new requirement to requirements.txt.
2. Update the environment with:
```bash
   conda env update -f environment.yml
```

### Environment Management

To manage the environment:

- Activate the environment:
```bash
  conda activate tactix-game-env
```

- Deactivate the environment:
```bash
  conda deactivate
```

### Testing

To run the tests, use the following commands from the project root:

- To run a specific test file (e.g., test_logic.py):
```bash
  python -m unittest tests/test_logic.py
```
- To run all tests:
```bash
  python -m unittest discover -s tests
```