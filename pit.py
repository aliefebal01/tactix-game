from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from tactix.tactixMove import Move
from agents.mcts.mcts_node import MCTSNode
from agents.mcts.mcts_agent_negamax import MCTSAgent_negamax
import numpy as np

def play_game():
    # Initialize the game and agent
    game = TactixGame()
    agent = MCTSAgent_negamax(player = 1, iterations=6000, exploration_weight=0.2)
    current_node = MCTSNode(game)
    

    print("Game started!")

    while current_node.state.getGameEnded() is None:
        if current_node.state.current_player == agent.player:
            # Agent's Turn
            current_node.state.display()
            print("\n Agent is making a move...")
            best_node = agent.best_child(current_node) # Get agent's best node
            current_node = best_node
            

        else:
            # Opponent's Turn
            current_node.state.display()
            print("\n Your Turn!")
            # Loop till the human makes a valid move
            valid_input = False
            while not valid_input:
                try:
                    # Take move input from the user (row, col, count, ver) in the format (0,1,2,0)
                    move_input = input("Enter your move as (row,col,count,ver): ").strip()

                    # Remove parentheses and split by commas
                    move_input = move_input.strip('()')  # Remove the surrounding parentheses
                    row, col, count, ver = map(int, move_input.split(','))  # Split by commas

                    # Creating the Move object from the input
                    human_move = Move(row, col, count, bool(ver))

                    # Check if the move is valid
                    current_node.state.base_board.is_valid_move(human_move)
                    valid_input = True
                except ValueError as e:
                    print(f"Invalid move: {e}. Please try again!")
            
            # Applying the move to get to the next state
            current_node.state = current_node.state.getNextState(human_move)
            
            

        # Announce winner 
    winner = current_node.state.getGameEnded().winner
    if winner == 1:
        print("Agent wins!")
    else:
        print("You win!")

if __name__ == "__main__":
    play_game()
            