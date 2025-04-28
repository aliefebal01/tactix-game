import torch
from tactix.utils import encode_action, decode_action
from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent_negamax import MCTSAgent_negamax
from agents.mcts.mcts_node import MCTSNode

import numpy as np

class TactixEnvironment:
    def __init__(self, mcts_iteration, mcts_lr, board_size=5):
        """Initialize the environment with a game instance."""
        self.game = TactixGame(height=board_size, width=board_size)  # Initialize the game logic
        self.state = None  # Current game state
        self.done = False  # Flag to indicate if the game is over
        self.starting_player = 1
        self.mcts_agent = MCTSAgent_negamax(player=-1, iterations=mcts_iteration, lr=mcts_lr)

    def reset(self):
        """Reset the environment to the initial state."""
        self.starting_player = -1 if self.starting_player == 1 else 1
        self.game = TactixGame(current_player=self.starting_player)  # Create a new instance of TactixGame

        if self.starting_player == -1: #MCTS agent plays first if it is the starting player
            game_node = MCTSNode(self.game)
            mcts_next_state = self.mcts_agent.best_child(game_node)
            self.game = mcts_next_state.state

        self.state = self.game.getPieces()  # Initialize the board state
        self.done = False  # Reset the game-over flag
        valid_moves_mask = self._generate_valid_moves_mask()
        return self._get_observation(), valid_moves_mask

    def step(self, action):
        """Execute the action in the environment."""
        move = decode_action(action, self.game.height)  # Decode action index
        self.game.makeMove(move)  # Execute the move
        game_ended = self.game.getGameEnded()

        if game_ended and game_ended.is_ended:
            reward = -1
            self.done = True
        else:
            game_node = MCTSNode(self.game)
            mcts_next_state = self.mcts_agent.best_child(game_node)
            self.game = mcts_next_state.state

            game_ended = self.game.getGameEnded()
            if game_ended and game_ended.is_ended:
                reward = 1
                self.done = True
            else:
                reward = 0

        self.state = self.game.getPieces()
        valid_moves_mask = self._generate_valid_moves_mask()
        return self._get_observation(), reward, self.done, valid_moves_mask

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