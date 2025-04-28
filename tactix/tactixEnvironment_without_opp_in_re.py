import torch
from tactix.utils import encode_action, decode_action
from tactix.tactixGame import TactixGame
import numpy as np

class TactixEnvironment:
    def __init__(self, board_size=5):
        """Initialize the environment with a game instance."""
        self.game = TactixGame(height=board_size,width=board_size)  # Initialize the game logic
        self.state = None  # Current game state
        self.done = False  # Flag to indicate if the game is over
        self.starting_player = 1

    def reset(self):
        """Reset the environment to the initial state."""
        self.starting_player = -1 if self.starting_player == 1 else 1
        self.game = TactixGame(height=self.game.height,width=self.game.height, current_player=self.starting_player)  # Create a new instance of TactixGame
        self.state = self.game.getPieces()  # Initialize the board state
        self.done = False  # Reset the game-over flag
        valid_moves_mask = self._generate_valid_moves_mask()
        return self._get_observation(), valid_moves_mask

    def step(self, action):
        """Execute the action in the environment."""
        move = decode_action(action, self.game.height)  # Decode action index
        self.game.makeMove(move)  # Execute the move
        self.state = self.game.getPieces()
        
        reward = 0
        # intermediate rewards for handling the sparseness of the rewards 
        existing_shapes = self.game.detect_all_shapes()
       
        if not existing_shapes['line_2'] and np.sum(self.game.getPieces()) % 2 == 0: # leaving not adjacent and even number of pieces to the opponent
            reward = -0.3
        
        
        if not existing_shapes['line_2'] and np.sum(self.game.getPieces()) % 2 != 0: # leaving not adjacent and odd number of pieces to the opponent
            reward = 0.3
        
        if self.game.height == 7:
            if existing_shapes['line_7'] and np.sum(self.game.getPieces()) == 7:
                reward = -0.6
        
        if self.game.height == 6:
            if existing_shapes['line_6'] and np.sum(self.game.getPieces()) == 6:
                reward = -0.6
        if self.game.height == 5:
            if existing_shapes['line_5'] and np.sum(self.game.getPieces()) == 5:
                reward = -0.6
        if self.game.height == 4:
            if existing_shapes['line_4'] and np.sum(self.game.getPieces()) == 4:
                reward = -0.6
            
        if existing_shapes['square'] and np.sum(self.game.getPieces()) == 4:
            reward = 0.3
        elif existing_shapes['triangle'] and np.sum(self.game.getPieces()) == 3:
            reward = -0.6
        elif existing_shapes['line_3'] and np.sum(self.game.getPieces()) == 3:
            reward = -0.6
        elif existing_shapes['line_2'] and np.sum(self.game.getPieces()) == 2:
            reward = -0.6

        # Game is ended or definitely won         
        game_ended = self.game.getGameEnded()
        if game_ended and game_ended.is_ended:
            reward = -1
            self.done = True
        elif np.sum(self.game.getPieces()) == 1:
            reward = 1
        

        return self._get_observation(), reward, self.done

    def _generate_valid_moves_mask(self):
        valid_moves = self.game.getValidMoves()
        valid_moves_mask = torch.zeros(self.game.height ** 3)
        for move in valid_moves:
            action_index = encode_action(move, self.game.height)
            valid_moves_mask[action_index] = 1
        return valid_moves_mask

    def _get_observation(self):
        """
        Convert the current board state into a PyTorch tensor.

        Returns:
            torch.Tensor: The current state as a tensor.
        """
        return torch.from_numpy(np.array(self.state, dtype=np.float32))

    def render(self):
        """Display the current board and game status."""
        self.game.display()