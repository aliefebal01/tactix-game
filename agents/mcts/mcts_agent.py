import numpy as np
from .mcts_node import MCTSNode


class MCTSAgent:
    def __init__(self, game, iterations=1000, exploration_weight=1 / np.sqrt(2)):
        self.game = game                                # the game object
        self.iterations = iterations                    # number of iterations per move 
        self.exploration_weight = exploration_weight    # exploration weight
