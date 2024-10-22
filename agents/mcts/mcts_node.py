import numpy as np
import random 

from tactix.tactixGame import TactixGame



LEARNING_PARAM = 1 / np.sqrt(2) # this could also be sqrt(2) chosen according to kocsis and szepesvari 2006


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state # the state of the board
        self.parent = parent # the parent node
        self.children = [] # list of the children nodes
        self.visits = 0 # number of visits
        self.wins = 0 # number of wins
        self.untried_actions = state.getValidMoves() # gets the valid moves from tactixGame.py 

        def is_fully_expanded(self):
            "Returns True if all actions have been tried."
            return len(self.untried_actions) == 0
        
        
        def expand(self):
            "Expand the node by adding a randomly choosen child node."
            idx = random.randrange(len(self.untried_actions))
            action = self.untried_actions.pop(idx)
            
            next_state = self.state.getNextState(action)
            child_node = MCTSNode(next_state, parent=self)
            self.children.append(child_node)
            return child_node
        
        def backpropagate(self, result):
            "Update this node - one node with the result of a simulation."
            self.visits += 1
            self.wins += result
            if self.parent:
                self.parent.backpropagate(result)
        


