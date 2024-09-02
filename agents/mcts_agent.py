import numpy as np


LEARNING_PARAM = 1 / np.sqrt(2) # this could also be sqrt(2) 


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
        
        def best_child(self, c_param=LEARNING_PARAM):
            "Return the child with the highest UCB score."
            choices_weights = [
                (c.wins / c.visits) + c_param * np.sqrt((2 * np.log(self.visits) / c.visits))
                for c in self.children
            ]
            return self.children[np.argmax(choices_weights)]
        
        def expand(self):
            "Expand the node by adding all possible child nodes."
            action = self.untried_actions.pop()
            next_state = self.state.getNextState(action)
            child_node = MCTSNode(next_state, parent=self)
            self.children.append(child_node)
            return child_node
        

class MCTSAgent:
    def __init__(self, game, iterations=1000, exploration_weight=1 / np.sqrt(2)):
        self.game = game                                # the game object
        self.iterations = iterations                    # number of iterations per move 
        self.exploration_weight = exploration_weight    # exploration weight

