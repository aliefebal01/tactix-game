from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from agents.mcts.mcts_agent_negamax import MCTSAgent_negamax
from tactix.tactixMove import Move
from agents.mcts.mcts_node import MCTSNode
import numpy as np



def agents_play(starting_player):
    
    game = TactixGame(current_player=starting_player)
    agent_10000 = MCTSAgent_negamax(player=1, iterations=15, exploration_weight=0.3)
    agent_30000 = MCTSAgent_negamax(player=-1, iterations=10, exploration_weight=0.4)
    current_node = MCTSNode(game)

    while current_node.state.getGameEnded() is None:
        if current_node.state.current_player == agent_10000.player:
            # Agent 10000's turn
            best_node = agent_10000.best_child(current_node)
            current_node = best_node
        else:
            # Agent 30000's turn
            best_node = agent_30000.best_child(current_node)
            current_node = best_node

    # Determine and return the winner
    winner = current_node.state.getGameEnded().winner
    return winner

def run_simulation(num_games=100):
    agent_10000_wins = 0
    agent_30000_wins = 0

    for i in range(num_games):
        starting_player = 1 if i % 2 == 0 else -1
        print(f"Running game {i + 1}...")
        winner = agents_play(starting_player)

        if winner == 1:
            agent_10000_wins += 1
            print("Agent 10000 wins this game.")
        elif winner == -1:
            agent_30000_wins += 1
            print("Agent 30000 wins this game.")

    print("\nSimulation complete!")
    print(f"Agent 10000 won {agent_10000_wins} times.")
    print(f"Agent 30000 won {agent_30000_wins} times.")

if __name__ == "__main__":
    run_simulation(100)