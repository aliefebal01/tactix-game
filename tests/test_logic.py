import unittest
import numpy as np
from tactix.tactixLogic import Board
from tactix.tactixMove import Move

class TestBoard(unittest.TestCase):

    def setUp(self):
        """Set up a standard 5x5 board for testing."""
        self.board = Board()

    def test_initial_board(self):
        """Test the initial board setup."""
        expected_pieces = np.ones((5, 5), dtype=int)
        np.testing.assert_array_equal(self.board.np_pieces, expected_pieces)
        self.assertEqual(self.board.current_player, 1)
        self.assertFalse(self.board.get_win_state().is_ended)

    def test_switch_player(self):
        """Test switching players."""
        self.assertEqual(self.board.current_player, 1)
        self.board.switch_player()
        self.assertEqual(self.board.current_player, -1)
        self.board.switch_player()
        self.assertEqual(self.board.current_player, 1)

    def test_is_valid_move_horizontal(self):
        """Test a valid horizontal move."""
        move = Move(row=0, col=0, piece_count=2, ver=False)
        self.assertTrue(self.board.is_valid_move(move))

    def test_is_valid_move_vertical(self):
        """Test a valid vertical move."""
        move = Move(row=0, col=0, piece_count=2, ver=True)
        self.assertTrue(self.board.is_valid_move(move))

    def test_invalid_move_out_of_bounds(self):
        """Test an invalid move that's out of bounds."""
        move = Move(row=4, col=4, piece_count=2, ver=True)
        with self.assertRaises(ValueError):
            self.board.is_valid_move(move)

    def test_invalid_move_over_empty(self):
        """Test an invalid move that passes over an empty cell."""
        self.board.np_pieces[1][0] = 0
        move = Move(row=0, col=0, piece_count=2, ver=True)
        with self.assertRaises(ValueError):
            self.board.is_valid_move(move)

    def test_remove_pieces_horizontal(self):
        """Test removing pieces horizontally."""
        move = Move(row=0, col=0, piece_count=2, ver=False)
        self.board.remove_pieces(move)
        expected_pieces = np.array([[0, 0, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(self.board.np_pieces, expected_pieces)

    def test_remove_pieces_vertical(self):
        """Test removing pieces vertically."""
        move = Move(row=0, col=0, piece_count=2, ver=True)
        self.board.remove_pieces(move)
        expected_pieces = np.array([[0, 1, 1, 1, 1],
                                    [0, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(self.board.np_pieces, expected_pieces)

    def test_get_win_state(self):
        """Test if the win state is correctly identified."""
        # Simulate a winning state by removing all pieces
        self.board.np_pieces = np.zeros((5, 5), dtype=int)
        self.board.win_state = self.board.get_win_state()
        self.assertTrue(self.board.win_state.is_ended)
        self.assertEqual(self.board.win_state.winner, 1) ## the board is initiated with player 1 and since there are no moves made player 1 wins 

    def test_valid_moves(self):
        """Test generation of valid moves."""
        moves = self.board.valid_moves()
        self.assertTrue(len(moves) > 0)  # Should return some valid moves
        for move in moves:
            self.assertTrue(self.board.is_valid_move(move))

    def test_with_np_pieces(self):
        """Test board copy with new pieces."""
        new_pieces = np.zeros((5, 5), dtype=int)
        new_board = self.board.with_np_pieces(np_pieces=new_pieces)
        np.testing.assert_array_equal(new_board.np_pieces, new_pieces)
        self.assertEqual(new_board.current_player, self.board.current_player)

    def test_reset_game(self):
        """Test resetting the game state."""
        self.board.np_pieces = np.zeros((5, 5), dtype=int)
        self.board.reset_game()
        expected_pieces = np.ones((5, 5), dtype=int)
        np.testing.assert_array_equal(self.board.np_pieces, expected_pieces)
        self.assertFalse(self.board.get_win_state().is_ended)


if __name__ == '__main__':
    unittest.main()