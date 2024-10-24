import random 
from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from tactix.tactixMove import Move

def play_game():
    # Initialize game and agent 
    game = TactixGame()             #Initializing the game on a 5x5 board with starting player 1 when not specified
    agent = MCTSAgent(player=1)     #Initializing the agent with player 1 and 1000 iterations per move, exploration weight is 1/sqrt(2)
    current_state = game
    

    print("Game started!")

    while current_state.getGameEnded() is None:

        if current_state.current_player == 1:
            # Agent's turn
            print("\n Agent is making a move...")
            best_node = agent.best_child(current_state) # Get agent's best move 
            current_state = best_node.state
            current_state.display()

        
        else:
            # Human's Turn
            print("\n Your Turn!(Player -1)")
            
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
                    current_state.base_board.is_valid_move(human_move)
                    valid_input = True
                except ValueError as e:
                    print(f"Invalid move: {e}. Please try again!")


            # Applying the move to get to the next state
            current_state = current_state.getNextState(human_move)
            current_state.display()
            
            # Updating the agent's root with the new state after Opponent's move
            agent.update_root(best_node, human_move)


    # Announce winner 
    winner = current_state.getGameEnded().winner
    if winner == 1:
        print("Agent wins!")
    else:
        print("You win!")

if __name__ == "__main__":
    play_game()