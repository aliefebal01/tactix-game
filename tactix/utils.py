
from tactix.tactixMove import Move

def encode_action(move, n):
    """Convert a Move object to an action index."""
    
    if move.ver:
        return int(n * sum(k for k in range(1, n+1)) + move.col * sum(k for k in range(1, n)) + move.row*(n - 0.5 - (move.row * 0.5)) + move.piece_count - 2)   
    else:
        return int(move.row * sum(k for k in range(1, n+1)) + (move.col * (n + 0.5 - (move.col * 0.5))) + move.piece_count - 1)  
    

def decode_action(action_index, n):
    """
    Invert the encoding scheme:
      - If action_index < n * sum_{k=1}^n(k) = 75 for n=5, it is a horizontal move.
      - Otherwise it is a vertical move (shifted by 75).

    Returns:
      Move(row, col, piece_count, ver)
    """
    # Some helpers
    S1 = sum(range(1, n+1))  # For n=5, S1 = 15
    S2 = sum(range(1, n))    # For n=5, S2 = 10

    # Decide horizontal vs vertical
    if action_index < n * S1:
        # -----------------
        # Horizontal moves
        # -----------------
        # Each row has 15 possible horizontal “sub-indices”.
        #   row part:   row = action_index // 15
        #   partial:    partial_index = action_index % 15
        ver = False

        row = action_index // S1
        partial_index = action_index % S1

        # Decode (col, piece_count) from partial_index in [0..14]
        # Buckets for partial_index:
        #   0..4   => col=0,  piece_count=(partial_index + 1)        (5 ways)
        #   5..8   => col=1,  piece_count=(partial_index - 4)        (4 ways)
        #   9..11  => col=2,  piece_count=(partial_index - 8)        (3 ways)
        #   12..13 => col=3,  piece_count=(partial_index - 11)       (2 ways)
        #   14     => col=4,  piece_count=1                         (1 way)

        if partial_index < 5:
            col = 0
            piece_count = partial_index + 1
        elif partial_index < 9:
            col = 1
            piece_count = partial_index - 4
        elif partial_index < 12:
            col = 2
            piece_count = partial_index - 8
        elif partial_index < 14:
            col = 3
            piece_count = partial_index - 11
        else:
            col = 4
            piece_count = 1

    else:
        # ---------------
        # Vertical moves
        # ---------------
        # They start at index = n*S1 = 75 for n=5.
        # So subtract 75 to get a 0-based index for vertical moves.
        #   i = action_index - 75
        #   col part:  col = i // 10   (since S2=10 for n=5)
        #   partial:   p  = i % 10
        # Then we decode (row, piece_count) out of that partial.
        ver = True

        offset = action_index - n * S1  # e.g. offset in [0..49] for n=5

        col = offset // S2
        partial_index = offset % S2

        # Now partial_index = row*(some-bucket-size) + (piece_count - 2).
        # In your code:
        #   row -> [0..4], but effectively [0..3] for piece_count≥2.
        #   piece_count in [2..(5 - row)], single-piece verticals are disallowed.
        #
        # Let’s decode that with simple piecewise checks:
        #   row=0 => offset for row-part=0
        #            piece_count -2 in [0..3]  => partial_index in [0..3]
        #   row=1 => offset for row-part=4
        #            piece_count -2 in [0..2]  => partial_index in [4..6]
        #   row=2 => offset for row-part=7
        #            piece_count -2 in [0..1]  => partial_index in [7..8]
        #   row=3 => offset for row-part=9
        #            piece_count -2 in [0..0]  => partial_index in [9]
        #   row=4 => offset for row-part=10
        #            but picking 2 vertically from row=4 is actually invalid
        #            so it typically won't appear in valid moves
        #
        # We still do the piecewise decode:

        if partial_index < 4:  
            # row=0, piece_count in 2..5
            row = 0
            piece_count = partial_index + 2
        elif partial_index < 7 and partial_index >= 4:
            # row=1, piece_count in [2..4]
            row = 1
            piece_count = partial_index - 4 + 2   # subtract the offset=4
        elif partial_index < 9 and partial_index >= 7:
            # row=2, piece_count in [2..3]
            row = 2
            piece_count = partial_index - 7 + 2   # offset=7
        elif partial_index < 10 and partial_index >= 9:
            # row=3, piece_count=2
            row = 3
            piece_count = partial_index - 9 + 2   # offset=9
        else:
            # row=4 => normally invalid for 2+ vertical picks,
            # but we'll decode it if it shows up. Typically won't
            # appear in the actual valid-move list if you are filtering out invalid moves.
            row = 4
            piece_count = partial_index - 10 + 2  # offset=10

    return Move(row=row, col=col, piece_count=piece_count, ver=ver)