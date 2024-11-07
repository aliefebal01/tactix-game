import numpy as np
import random 
from .mcts_node import MCTSNode

LEARNING_PARAM = 1 / np.sqrt(2) # this could also be sqrt(2) chosen according to kocsis and szepesvari 2006
DEFAULT_ITERATIONS = 30000

class MCTSAgent:
    def __init__(self, player, iterations=None, exploration_weight=None):
        self.player = player                                              # the player
        self.iterations = iterations or DEFAULT_ITERATIONS                # number of iterations per move (during training)
        self.exploration_weight = exploration_weight or LEARNING_PARAM    # exploration weight
        self.root = None                                                  # the root node of the tree

    def best_child(self, passed_node):
        "Run MCTS from the current root and select the best move."
        
        self.set_root(passed_node) # Set the root node to the current state
        
        for _ in range(self.iterations):
            node = self.tree_policy(self.root)
            result = self.rollout(node)
            node.backpropagate(result)
    
        children_ucb = [
            (child.wins / child.visits) + self.exploration_weight * np.sqrt(2 * np.log(node.visits) / child.visits)
            for child in self.root.children
        ]
        for x, child in enumerate(self.root.children):
            print(f"Child{x+1}: visits: {child.visits}, wins: {child.wins}, ucb: {children_ucb[x]}")

        
        best_node = self.ucb(self.root, c_param=0)

        for x, child in enumerate(self.root.children):
            if child.state == best_node.state:
                print(f"Child Chosen:{x+1} visits: {child.visits}, wins: {child.wins}, ucb: {children_ucb[x]}")
                break

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
        

    def set_root(self, node):
        "Set the root node "
        if node.parent is None: 
            # Initial game state: set the node directly as the root
            self.root = node
        else:
            # Attempt to locate a matching descendant for node within current root's tree
            found_node = self.find_matching_descendant(node)

            if found_node is not None:
                # Matching node found in the current tree, so set it as the new root
                self.root = found_node
            else:
                # Matching node not found: create a new root with the target state
                self.root = MCTSNode(node.state)
    

    def find_matching_descendant(self, target_node):
        "Searching the grandchildren of the current node to find the target node"
        # If no root has been set (first call by the second player), return None to signal this state.
        if self.root == None:
            return None

        # Search through the children and grandchildren of the current root for a matching state.
        for child in self.root.children:
            for grand_child in child.children:
                if grand_child.state == target_node.state:
                    return grand_child
        # If no matching node is found, return None to signal the need for a new root.
        return None 
            




