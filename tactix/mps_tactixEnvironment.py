import torch
from tactix.utils import encode_action, decode_action
from tactix.tactixGame import TactixGame
import numpy as np
import time

class TactixEnvironment():
    def __init__(self, ):
        """Initialize the environment with a game instance."""
        self.game = TactixGame()  # Initialize the game logic
        self.state = None  # Current game state
        self.done = False  # Flag to indicate if the game is over
        self.starting_player = 1
        self.device = 'mps'

    def reset(self):
        """Reset the environment to the initial state."""
        self.starting_player = -1 if self.starting_player == 1 else 1
        self.game = TactixGame(current_player=self.starting_player)  # Create a new instance of TactixGame
        self.done = False  # Reset the game-over flag

        # Get the board as a numpy array, convert once to a [1, state_size] MPS Tensor
        pieces = self.game.getPieces()       # shape (5,5) as np array
        pieces = pieces.reshape(1, -1)       # now shape [1, 25]
        self.state = torch.from_numpy(pieces).float().to(self.device)

        valid_moves_mask = self._generate_valid_moves_mask().unsqueeze(0)

        return self.state, valid_moves_mask

    def step(self, action):
        """Execute the action in the environment."""
        move = decode_action(action, self.game.height)  # Decode action index
        self.game.makeMove(move)  # Execute the move cpu-based
        game_ended = self.game.getGameEnded()

        # Decide reward on CPU side (just a float), then convert once to MPS
        if game_ended and game_ended.is_ended:
            reward = -1.0
            self.done = True
        else:
            # Opponent's random move
            opponent_move = self.game.get_random_move()
            self.game.makeMove(opponent_move)
            game_ended = self.game.getGameEnded()

            if game_ended and game_ended.is_ended:
                reward = 1.0
                self.done = True
            else:
                reward = 0.0

        # Convert reward to a scalar on MPS only once
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Update self.state
        pieces = self.game.getPieces()          # shape (5,5)
        pieces = pieces.reshape(1, -1)          # [1, 25]
        self.state = torch.from_numpy(pieces).float().to(self.device)
        
        valid_moves_mask = self._generate_valid_moves_mask().unsqueeze(0)

        return self.state, reward, self.done, valid_moves_mask

    def _generate_valid_moves_mask(self):
        """Return a 1D mask of shape [125] on MPS."""
        valid_moves = self.game.getValidMoves()
        mask = torch.zeros(125, dtype=torch.float32, device=self.device)
        for move in valid_moves:
            action_index = encode_action(move, self.game.height)
            mask[action_index] = 1.0
        return mask

    def _get_observation(self):
        """
        Convert the current board state into a PyTorch tensor.

        Returns:
            torch.Tensor: The current state as a tensor.
        """
        return self.state

    def render(self):
        """Display the current board and game status."""
        self.game.display()