from collections import namedtuple 
from tactix.tactixMove import Move
import numpy as np



DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = 5

WinState = namedtuple('WinState', 'is_ended winner')


class Board():

    """Tactix Board"""

    def __init__(self, height=None, width=None, np_pieces=None, current_player=1):
        "Set up initial board configuration."
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.current_player = current_player  # Player 1 starts if it is not specified

        if np_pieces is None:
            self.np_pieces = np.ones([self.height, self.width], dtype=int)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

        
        self.win_state = WinState(is_ended=False, winner=None)

    def switch_player(self):
        self.current_player = -1 if self.current_player == 1 else 1 # switching among players 1 and -1

    
    def is_valid_move(self, move):  
        "Check if move is valid."
        
        if (move.row < 0 or move.row >= self.height):
            raise ValueError("Can't play; selected row is outside the board")
        
        if (move.col < 0 or move.col >= self.width):
            raise ValueError("Can't play; selected column is outside the board")
        
        if move.ver is True:
            "When selecting vertical cells"
            if move.row + move.piece_count > self.height:
                raise ValueError("Can't play; selected cells are outside the board")

            for i in range(move.row, move.row + move.piece_count):
                if self.np_pieces[i][move.col] == 0:
                    raise ValueError("Can't play; selected cells have empty cells")
        else:
            "When selecting horizontal cells"

            if move.col + move.piece_count > self.width:
                raise ValueError("Can't play; selected cells are outside board")

            for i in range(move.col, move.col + move.piece_count):
                if self.np_pieces[move.row][i] == 0:
                    raise ValueError("Can't play; selected cells have empty cells")
        
        
                

    def remove_pieces(self, move):
        "Remove stone from board."
        
        self.is_valid_move(move)

        
        if move.ver is True:
            "removing pieces vertically"
            for i in range(move.row, move.row + move.piece_count):
                self.np_pieces[i][move.col] = 0
        else:
            "removing pieces horizontally"
            for i in range(move.col, move.col + move.piece_count):
                self.np_pieces[move.row][i] = 0
        
        # Player is switched in the game class after the move is made

        
    def valid_moves(self):

        valid_moves = [] # list of valid moves
        
        # Checking for Horizontal moves
        for i in range(self.height):
            count = 0
            for j in range(self.width):
                if self.np_pieces[i][j] == 1:
                    count += 1
                    move = Move(row=i, col=j, piece_count=1) # picking one piece / not necessary to specify ver variable / 1 piece only
                    valid_moves.append(move)
                    if count > 1: # picking more then one piece
                        temp_count = count
                        while temp_count > 1:
                            move = Move(row=i, col=j-temp_count+1, piece_count=temp_count, ver=False)
                            valid_moves.append(move)
                            temp_count -= 1
                else:
                    count = 0


        # Checking for Vertical moves
        for i in range(self.width):
            count = 0
            for j in range(self.height):
                if self.np_pieces[j][i] == 1:
                    count += 1
                    # one piece moves are only in horizontal counted to prevent duplicates
                    if count > 1:
                        temp_count = count
                        while temp_count > 1:
                            move = Move(row=j-temp_count+1, col=i, piece_count=temp_count, ver=True)
                            valid_moves.append(move) 
                            temp_count -= 1
                else:
                    count = 0

        return valid_moves
    


    def with_np_pieces(self, np_pieces, current_player=None):
        """Create copy of board with specified pieces."""
        
        if current_player is None:
            current_player = self.current_player
        
        if np_pieces is None:
            np_pieces = self.np_pieces
        return Board(self.height, self.width, np_pieces, current_player)
    

    def get_win_state(self):
        # Check if the board is empty
        if np.all(self.np_pieces == 0):
            self.win_state = WinState(is_ended=True, winner= self.current_player) # if the board is empty the current player wins since the current player switches after every move made
        return self.win_state
        

    def reset_game(self):
        "Reset the game state to the initial configuration."
        self.np_pieces = np.ones([self.height, self.width], dtype=int)
        self.win_state = WinState(is_ended=False, winner=None)
    

    def __str__(self):
        return str(self.np_pieces)
                    

        
         