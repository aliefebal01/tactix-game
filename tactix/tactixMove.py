class Move():
    
    
    def __init__(self, row, col, piece_count, ver=False):  
        self.row = row
        self.col = col
        self.piece_count = piece_count
        self.ver = ver

    def __repr__(self):
        return f"[{self.row}, {self.col}, {self.piece_count}, {self.ver}]"
    
    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.row == other.row and self.col == other.col and
                    self.piece_count == other.piece_count and self.ver == other.ver)
        return False

    def __hash__(self):
        return hash((self.row, self.col, self.piece_count, self.ver))