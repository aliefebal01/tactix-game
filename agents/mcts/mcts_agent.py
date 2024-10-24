import numpy as np
import random 
from .mcts_node import MCTSNode

LEARNING_PARAM = 1 / np.sqrt(2) # this could also be sqrt(2) chosen according to kocsis and szepesvari 2006


class MCTSAgent:
    def __init__(self, player, iterations=30000, exploration_weight=None):
        self.player = player                                              # the player
        self.iterations = iterations                                      # number of iterations per move (during training)
        self.exploration_weight = exploration_weight or LEARNING_PARAM    # exploration weight
        self.root = None                                                  # the root node of the tree

    def best_child(self, state):
        "Run MCTS from the current root and select the best move."
        
        self.root = MCTSNode(state)
        
        for _ in range(self.iterations):
            node = self.tree_policy(self.root)
            result = self.rollout(node)
            node.backpropagate(result)
    
        best_node = self.ucb(self.root, c_param=0)
        return best_node
    
    def ucb(self, node, c_param=None):
        "Return the child with the highest UCB score."
        if c_param is None:
            c_param = self.exploration_weight

        children_ucb = [
            (child.wins / child.visits) + c_param * np.sqrt(2 * np.log(node.visits) / child.visits)
            for child in node.children
        ]
        
        return node.children[np.argmax(children_ucb)]
    
    
    def tree_policy(self, node): #Traversing down the tree to a leaf node
        "Select a node to explore or expand" 
        while node.state.getGameEnded() is None: #Stop if the game is over
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = self.ucb(node)
        return node
    

    def rollout(self, node):
        "Simulate a random playout from the current node."
        current_state = node.state
        while current_state.getGameEnded() is None:  # Playing until the game ends 
            action = random.choice(current_state.getValidMoves())          
            current_state = current_state.getNextState(action)    
        winner = current_state.getGameEnded().winner           # Returning the winner (-1 or 1)
        
        if winner == self.player:   # Returning the reward of the game
            return 1
        else:
            return -1 
        

    def update_root(self, best_node, opponent_move):
        """
        Update the root node after the opponent makes a move, using the best node
        selected by the agent and applying the opponent's move.
        """
        # Get the next state after applying the opponent's move to the best_node
        new_state = best_node.state.getNextState(opponent_move)
        
        # Look for the child node of the best_node that matches the new state
        found = False
        for child in best_node.children:
            if child.state == new_state:
                print(f"Updating root to new state after opponent's move.")
                self.root = child  # Update the root to the corresponding child node
                self.root.parent = None  # Detach from the previous tree to free up memory
                found = True
                break

        # If the opponent's move wasn't in the tree, create a new root node
        if not found:
            self.root = MCTSNode(new_state)  # Create a new root with the opponent's move state
            

    



