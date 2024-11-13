import json
import numpy as np
from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from agents.mcts.mcts_node import MCTSNode

def agents_play(agent_1_params, agent_2_params, agent_1_name, agent_2_name):
    game = TactixGame()
    agent_1 = MCTSAgent(player=1, **agent_1_params)
    agent_2 = MCTSAgent(player=-1, **agent_2_params)
    current_node = MCTSNode(game)

    while current_node.state.getGameEnded() is None:
        if current_node.state.current_player == agent_1.player:
            # Agent 1's turn
            best_node = agent_1.best_child(current_node)
            current_node = best_node
        else:
            # Agent 2's turn
            best_node = agent_2.best_child(current_node)
            current_node = best_node

    # Determine and return the winner
    winner = current_node.state.getGameEnded().winner
    return winner

def run_simulation_set(num_games, learning_rates, iteration_counts, player_orders, output_file="simulation_results.json"):
    results = []

    for lr_1, lr_2 in learning_rates:
        for iter_1, iter_2 in iteration_counts:
            for player_order in player_orders:
                # Set agent configurations and names
                agent_1_name = f"Agent_{lr_1}_{iter_1}"
                agent_2_name = f"Agent_{lr_2}_{iter_2}"
                
                if player_order == "Agent 10000 First":
                    agent_1_params = {"iterations": iter_1, "exploration_weight": lr_1}
                    agent_2_params = {"iterations": iter_2, "exploration_weight": lr_2}
                else:
                    agent_1_params = {"iterations": iter_2, "exploration_weight": lr_2}
                    agent_2_params = {"iterations": iter_1, "exploration_weight": lr_1}
                    agent_1_name, agent_2_name = agent_2_name, agent_1_name  # Swap names if player order changes

                # Run a batch of games for the current configuration
                agent_1_wins = 0
                agent_2_wins = 0

                print(f"\nRunning simulation: {agent_1_name} vs {agent_2_name}")

                for i in range(num_games):
                    print(f"Running game {i + 1}...")
                    winner = agents_play(agent_1_params, agent_2_params, agent_1_name, agent_2_name)

                    if winner == 1:
                        agent_1_wins += 1
                        print(f"{agent_1_name} wins this game.")
                    elif winner == -1:
                        agent_2_wins += 1
                        print(f"{agent_2_name} wins this game.")

                # Record the results for this configuration
                results.append({
                    "agent_1_name": agent_1_name,
                    "iterations_1": iter_1,
                    "learning_rate_1": lr_1,
                    "agent_2_name": agent_2_name,
                    "iterations_2": iter_2,
                    "learning_rate_2": lr_2,
                    "player_order": player_order,
                    "agent_1_wins": agent_1_wins,
                    "agent_2_wins": agent_2_wins
                })

                print(f"Results: {agent_1_name} won {agent_1_wins} times, {agent_2_name} won {agent_2_wins} times.\n")

    # Save results to JSON file
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nAll simulation results saved to {output_file}")

if __name__ == "__main__":
    # Define different learning rates and iteration counts for testing
    learning_rates = [
    (0.1, 0.1),               # Very low exploration
    (0.2, 0.2),               # Low exploration
    (0.2, 1/np.sqrt(2)),      # Moderate exploration for second agent
    (0.3, 0.3),               # Moderate exploration for both
    (0.3, 0.5),               # Balanced low-medium exploration
    (1/np.sqrt(2), 1/np.sqrt(2)),  # Common MCTS exploration rate
    (0.4, 0.6),               # Slightly higher exploration
    (0.5, 0.7),               # Higher exploration for both
    (0.5, 1),                 # High exploration for second agent
    (1, 1)                    # Very high exploration
    ]
    iteration_counts = [2
    (1000, 1000),             # Baseline
    (1000, 3000),             # Uneven: second agent has more simulations
    (2000, 2000),             # Moderate iterations
    (2000, 5000),             # Uneven with higher value for second agent
    (3000, 3000),             # Higher for both agents
    (3000, 10000),            # Very uneven; test if iteration count advantage is critical
    (5000, 5000),             # Higher baseline
    (10000, 10000),           # High simulations for both agents
    (10000, 15000),           # Uneven with high count for second agent
    (15000, 15000)            # Stress test for very high iterations
    ]
    player_orders = ["Agent 10000 First", "Agent 30000 First"]

    # Run simulation set and save results to a JSON file
    run_simulation_set(num_games=100, learning_rates=learning_rates, iteration_counts=iteration_counts, player_orders=player_orders, output_file="simulation_results.json")