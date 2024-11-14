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
        action = self.untried_actions.pop(idx) #choosing a random action in O(n) time
            
        next_state = self.state.getNextState(action)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node
        
    def backpropagate(self, result):
        "Update this node - one node with the result of a simulation."
        self.visits += 1
        if result == 1:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

    def reverse_negamax(self, result):
        """
        Updates the win statistics for the current node and its ancestors in a negamax style, 
        where a loss for the current player is considered a win for the opponent.

        Parameters:
        - result: The final game result (player identifier) that indicates the winner of the simulation.
        """
        # Increment visit count to track the number of simulations reaching this node
        self.visits += 1

        # If the result matches the current player, it was a loss (reverse scoring for negamax)
        if result == self.state.current_player:
            self.wins -= 1
        else:
            self.wins += 1
        # Recursively apply negamax-style backpropagation up the tree to the parent node
        if self.parent:
            self.parent.reverse_negamax(result)

        


