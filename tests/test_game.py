import unittest
import numpy as np
from tactix.tactixGame import TactixGame
from tactix.tactixMove import Move

class TestTactixGame(unittest.TestCase):

    def setUp(self):
        """Set up the Tactix game for testing."""
        self.game5 = TactixGame()
        self.game6 = TactixGame(height=6, width=6)
        self.game7 = TactixGame(height=7, width=7)

    def test_get_init_board(self):
        """Test initial board state."""
        expected_pieces5 = np.ones((5, 5), dtype=int) # 5x5 board with all pieces
        np.testing.assert_array_equal(self.game5.getInitBoard(), expected_pieces5)

        expected_pieces6 = np.ones((6, 6), dtype=int) # 6x6 board with all pieces
        np.testing.assert_array_equal(self.game6.getInitBoard(), expected_pieces6)

        expected_pieces7 = np.ones((7, 7), dtype=int) # 7x7 board with all pieces
        np.testing.assert_array_equal(self.game7.getInitBoard(), expected_pieces7)

        

    def test_get_board_size(self):
        """Test board size."""
        self.assertEqual(self.game5.getBoardSize(), (5, 5))
        self.assertEqual(self.game6.getBoardSize(), (6, 6))
        self.assertEqual(self.game7.getBoardSize(), (7, 7))



    def test_get_action_size(self):
        """Test if action size is calculated correctly."""

        self.assertTrue(self.game5.getActionSize() > 0)  # Should return some valid moves

        self.assertTrue(self.game6.getActionSize() > 0)  # Should return some valid moves

        self.assertTrue(self.game7.getActionSize() > 0)  # Should return some valid moves



    def test_get_next_state(self):
        """Test the transition to the next state."""
        
        move = Move(row=0, col=0, piece_count=2, ver=False) #doing the same move on every board size
        
        pieces5 = self.game5.getInitBoard()
        next_pieces5 = self.game5.getNextState(pieces5, move)
        expected_pieces5 = np.array([[0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(next_pieces5, expected_pieces5)

        pieces6 = self.game6.getInitBoard()
        next_pieces6 = self.game6.getNextState(pieces6, move)
        expected_pieces6 = np.array([[0, 0, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(next_pieces6, expected_pieces6)

        pieces7 = self.game7.getInitBoard()
        next_pieces7 = self.game7.getNextState(pieces7, move)
        expected_pieces7 = np.array([[0, 0, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(next_pieces7, expected_pieces7)
       


    def test_get_valid_moves(self):
        """Test getting valid moves."""
        
        # For 5x5 board
        moves5 = self.game5.getValidMoves()
        print(f'5x5 board has {len(moves5)} different valid moves initially')
        self.assertTrue(len(moves5) > 0)

        # For 6x6 board 
        moves6 = self.game6.getValidMoves()
        print(f'6x6 board has {len(moves6)} different valid moves initially')
        self.assertTrue(len(moves6) > 0)

        # For 7x7 board
        moves7 = self.game7.getValidMoves()
        print(f'7x7 board has {len(moves7)} different valid moves initially')
        self.assertTrue(len(moves7) > 0)

    
    
    def test_get_game_ended(self):
        """Test checking if the game has ended."""
        
        # Create a list of games with different board sizes
        games = [(self.game5, 5), (self.game6, 6), (self.game7, 7)]
        
        for game, size in games:
            game = TactixGame(height=size, width=size) # initialized a game with the size of the board and player 1 starts the game 
            
            # Simulate clearing the board row by row
            for i in range(size):
                move = Move(row=i, col=0, piece_count=size, ver=False)
                game.makeMove(move)

            winstate = game.getGameEnded()
            
            if size % 2 == 0:  # if the board size is even then player 1 should win 
                self.assertEqual(game.getGameEnded(), (True , 1))
            else:              # if the board size is odd then player -1 should win
                self.assertEqual(game.getGameEnded(), (True , -1))
            
   
    def test_display(self):
        """Test displaying every board."""
        
        board5 = self.game5.getInitBoard()
        self.game5.display(board5) 

        board6 = self.game6.getInitBoard()
        self.game6.display(board6)

        board7 = self.game7.getInitBoard()
        self.game7.display(board7)


if __name__ == '__main__':
    unittest.main()
