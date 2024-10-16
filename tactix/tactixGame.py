import numpy as np
from tactix.tactixLogic import Board


class TactixGame():
    

    def __init__(self, height=None, width=None, np_pieces=None, current_player = 1):
        self.base_board = Board(height, width, np_pieces, current_player)  

    def getInitBoard(self):
        "Returns the pieces as a numpy array."
        return self.base_board.np_pieces
    
    def getBoardSize(self):
        "Returns the board size as a tuple."
        return self.base_board.height, self.base_board.width
    
    def getActionSize(self):
        "Returns the number of possible moves at current board state."
        return len(self.base_board.valid_moves())
    
    def getNextState(self, move):
        """Returns a copy of the next board state without altering the current state."""
        b = self.base_board.with_np_pieces()
        b.remove_pieces(move)
        b.switch_player()
        return b
    
    def makeMove(self, move):
        """Applies a move directly to the game board and updates the current player."""
        self.base_board.remove_pieces(move)
        self.base_board.switch_player()
    
    def getValidMoves(self):
        "returning the valid moves"
        return self.base_board.valid_moves()
    
    def getGameEnded(self):  #this function returns the reward and if the current player has no pieces to remove than the player before him lost the game 
        """
        Returns:
        - winstate if the game has ended -> (is_ended, winner)
        - None if the game has not ended
        """
        winstate = self.base_board.get_win_state()
        if winstate.is_ended:
            return winstate  # Returning the winstate object instead of a reward
        else:
            return None  # Or any other value indicating the game is not yet ended
        
        
   
    
    @staticmethod
    def display(board_pieces):
        print(" -----------------------")
        print(board_pieces)
        print(" -----------------------")