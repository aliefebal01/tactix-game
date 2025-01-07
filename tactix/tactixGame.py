from collections import namedtuple
import numpy as np
import random
from tactix.tactixMove import Move
from tactix.tactixLogic import Board

DEFAULT_STARTING_PLAYER = 1
DEFAULT_HEIGHT = 5
WinState = namedtuple('WinState', 'is_ended winner')


class TactixGame():
    

    def __init__(self, height=None, width=None, np_pieces=None, current_player = None):
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_HEIGHT
        self.base_board = Board(height, width, np_pieces if np_pieces is not None else None)
        self.current_player = current_player or DEFAULT_STARTING_PLAYER
        self.win_state = WinState(is_ended=False, winner=None)  # Initialize with is_ended=False

    def __eq__(self, other):
        if isinstance(other, TactixGame):
            return (self.base_board == other.base_board and self.current_player == other.current_player)
        return False 

    def getPieces(self):
        "Returns the pieces as a numpy array."
        return self.base_board.np_pieces
    
    def getBoardSize(self):
        "Returns the board size as a tuple."
        return self.base_board.height, self.base_board.width
    
    def switch_player(self):
        self.current_player = -1 if self.current_player == 1 else 1

    def getActionSize(self):
        "Returns the number of possible moves at current board state."
        return len(self.base_board.valid_moves())
    
    def getNextState(self, move):
        """Returns a copy of the next board state without altering the current state."""
        b = self.base_board.get_board_copy()
        b.remove_pieces(move)
        return TactixGame(height=b.height, width=b.width, np_pieces=b.np_pieces, current_player=self.current_player*-1)
    
    def makeMove(self, move):
        """Applies a move directly to the game board and updates the current player."""
        self.base_board.remove_pieces(move)
        self.switch_player()
    
    def getValidMoves(self):
        "returning the valid moves"
        return self.base_board.valid_moves()
    
    def getGameEnded(self):  #this function returns the reward and if the current player has no pieces to remove than the player before him lost the game 
        """
        Returns:
        - winstate if the game has ended -> (is_ended, winner)
        - None if the game has not ended
        """
        if self.base_board.get_board_empty():
            # If the board is empty, the current player wins (because the opponent made the last move)
            winner = self.current_player  # Since current_player switches after every move, current_player wins when the board is empty.
            return WinState(is_ended=True, winner=winner)
        else:
            return None  # Game is not over yet
        
    def get_dqn_reward(self):
        """
        Calculate the reward for the current board state in Tactix.

        Efficiently checks if the board has exactly one piece left using NumPy's sum function.
        Returns a reward of 1 if the condition is met.

        Returns:
            int: The reward for the current state (1 if the board has only 
            one piece left, otherwise None or no reward).
        """
        piece_count = np.sum(self.base_board.np_pieces)
        if piece_count == 1:
            return 1
        else:
            return 0
        
    def get_random_move(self):
        valid_moves = self.getValidMoves()
        if len(valid_moves) > 1:
            x = True
            while x:
                move = random.choice(valid_moves)
                if move.piece_count != np.sum(self.base_board.np_pieces):
                    x = False
                    return move
        else:
            return valid_moves[0]
        

    def encode_action(move):
        """Convert a Move object to an action index."""
        index = move.row * 25 + move.col * 5 + (move.piece_count - 1)
        if move.ver:
            index += 60  # Offset for vertical moves
        return index

    def decode_action(action_index):
        """Convert an action index back to a Move object."""
        ver = action_index >= 60
        if ver:
            action_index -= 60
        row = action_index // 25
        col = (action_index % 25) // 5
        piece_count = (action_index % 5) + 1
        return Move(row=row, col=col, piece_count=piece_count, ver=ver)

        
    def display(self):
        """Display the current board and current player."""
        print(" -----------------------")
        print(self.base_board.np_pieces)  # Display the board pieces
        print(" -----------------------")
        print(f"Current player: {self.current_player}")  # Display the current player

        # Check if the game has ended
        game_ended = self.getGameEnded()
        if game_ended and game_ended.is_ended:
            print(f"Game over! Winner: Player {game_ended.winner}")
        else:
            print("Game continues, no winner yet.") 