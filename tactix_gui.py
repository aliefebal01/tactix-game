import tkinter as tk
from tactix.tactixGame import TactixGame
from tactix.tactixMove import Move
from agents.mcts.mcts_agent import MCTSAgent

class TactixGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tactix Game")
        self.game = TactixGame()
        self.agent = MCTSAgent(player=1, iterations=10000)
        self.buttons = [[None for _ in range(5)] for _ in range(5)]  # 5x5 board buttons

        self.create_board()
        self.update_board()
        self.status_label = tk.Label(self.root, text="Your turn!")
        self.status_label.grid(row=6, column=0, columnspan=5)

    def create_board(self):
        for row in range(5):
            for col in range(5):
                btn = tk.Button(self.root, width=6, height=3,
                                command=lambda r=row, c=col: self.human_move(r, c))
                btn.grid(row=row, column=col)
                self.buttons[row][col] = btn

    def update_board(self):
        for row in range(5):
            for col in range(5):
                piece = self.game.getPieces()[row][col]
                self.buttons[row][col].config(text="X" if piece == 1 else "", state="normal" if piece == 1 else "disabled")

    def human_move(self, row, col):
        try:
            move = Move(row=row, col=col, piece_count=1, ver=False)  # Example move; modify to accept multiple pieces
            self.game.makeMove(move)
            self.update_board()
            if self.game.getGameEnded():
                self.end_game()
            else:
                self.agent_move()
        except ValueError as e:
            self.status_label.config(text=str(e))

    def agent_move(self):
        self.status_label.config(text="Agent's turn...")
        self.root.update()  # Force update to show status

        best_node = self.agent.best_child(self.game)
        self.game = best_node.state
        self.update_board()
        if self.game.getGameEnded():
            self.end_game()
        else:
            self.status_label.config(text="Your turn!")

    def end_game(self):
        winner = self.game.getGameEnded().winner
        self.status_label.config(text=f"Game Over! Winner: {'Agent' if winner == 1 else 'You'}")
        for row in range(5):
            for col in range(5):
                self.buttons[row][col].config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = TactixGameGUI(root)
    root.mainloop()