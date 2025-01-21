from collections import namedtuple 
from tactix.tactixMove import Move
import numpy as np

# switch_player ve get_win_state functions tactixgame e gececek get_win_state yerine get_board_empty winstate i de tactixgame e gececek

DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = 5




class Board():

    """Tactix Board"""

    def __init__(self, height=None, width=None, np_pieces=None):
        "Set up initial board configuration."
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        

        if np_pieces is None:
            self.np_pieces = np.ones([self.height, self.width], dtype=int)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)
            

    def __eq__(self, other):
        if isinstance(other, Board):
            return (self.height == other.height and self.width == other.width and
                    np.array_equal(self.np_pieces, other.np_pieces))
        return False
    

    def is_valid_move(self, move):  
        "Check if move is valid."

        # Convert integers 0 and 1 to booleans (True, False) if necessary
        if isinstance(move.ver, int):
            move.ver = bool(move.ver)
        
        # Check if ver is 0 (False) or 1 (True)
        if not isinstance(move.ver, bool):
            raise ValueError("The 'ver' parameter must be either 0 (horizontal) or 1 (vertical)")

        
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
    


    def get_board_copy(self):
        """Create a deep copy of the board object."""
        # Use np.copy to create a copy of the np_pieces array
        np_pieces_copy = np.copy(self.np_pieces)
    
        # Return a new Board instance with the copied pieces and same dimensions
        return Board(self.height, self.width, np_pieces_copy)
    

    def get_board_empty(self):
        # Check if the board is empty
        if np.all(self.np_pieces == 0):
            return True
        return False
        

    def reset_board(self):
        "Reset the game state to the initial configuration."
        self.np_pieces = np.ones([self.height, self.width], dtype=int)

    
    
    def generate_random_board(self, piece_count):

        if piece_count > self.height * self.width:
            raise ValueError("The number of pieces exceeds the total cells in the board.")
        
        # Create an empty board
        board = np.zeros((self.height, self.width), dtype=int)
        
        # Flatten the board to make placing pieces easier
        flat_board = board.flatten()
        
        # Choose random positions to place the pieces
        random_indices = np.random.choice(len(flat_board), size=piece_count, replace=False)
        
        # Set the chosen positions to 1
        flat_board[random_indices] = 1
        
        # Reshape back to the original board dimensions
        board = flat_board.reshape((self.height, self.width))

        self.np_pieces = board
        
    

    def __str__(self):
        return str(self.np_pieces)
                    

        
         