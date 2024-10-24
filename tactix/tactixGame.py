from collections import namedtuple
import numpy as np
from tactix.tactixLogic import Board

DEFAULT_STARTING_PLAYER = 1
WinState = namedtuple('WinState', 'is_ended winner')

class TactixGame():
    

    def __init__(self, height=None, width=None, np_pieces=None, current_player = None):
        self.base_board = Board(height, width, np_pieces if np_pieces is not None else None)
        self.current_player = current_player or DEFAULT_STARTING_PLAYER
        self.win_state = WinState(is_ended=False, winner=None)  # Initialize with is_ended=False

    def __eq__(self, other):
        if isinstance(other, TactixGame):
            return (self.base_board == other.base_board)
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