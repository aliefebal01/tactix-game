import json
import numpy as np
from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from agents.mcts.mcts_agent_negamax import MCTSAgent_negamax
from agents.mcts.mcts_node import MCTSNode

def agents_play(agent_1_params, agent_2_params):
    game = TactixGame()
    agent_1 = MCTSAgent_negamax(player=1, **agent_1_params)
    agent_2 = MCTSAgent_negamax(player=-1, **agent_2_params)
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

def run_simulations(iterations, learning_rates, num_games, output_file="simulation_results.json"):
    results = []

    for iter_1 in iterations:
        for iter_2 in iterations:
            # if iter_1 >= iter_2:
            #     continue  # Avoid duplicate and self-play

            for lr_1 in learning_rates:
                for lr_2 in learning_rates:
                    if lr_1 >= lr_2:
                        continue  # Avoid duplicate and self-play

                    agent_1_params = {"iterations": iter_1, "exploration_weight": lr_1}
                    agent_2_params = {"iterations": iter_2, "exploration_weight": lr_2}

                    agent_1_wins = 0
                    agent_2_wins = 0

                    print(f"\nRunning simulation: Agent_{iter_1}_{lr_1} vs Agent_{iter_2}_{lr_2}")

                    for i in range(num_games):
                        print(f"Running game {i + 1}...")
                        winner = agents_play(agent_1_params, agent_2_params)

                        if winner == 1:
                            agent_1_wins += 1
                            print(f"Agent_{iter_1}_{lr_1} wins this game.")
                        elif winner == -1:
                            agent_2_wins += 1
                            print(f"Agent_{iter_2}_{lr_2} wins this game.")

                    # Record the results for this configuration
                    results.append({
                        "agent_1": f"Agent_{iter_1}_{lr_1}",
                        "agent_2": f"Agent_{iter_2}_{lr_2}",
                        "agent_1_wins": agent_1_wins,
                        "agent_2_wins": agent_2_wins
                    })

                    print(f"Results: Agent_{iter_1}_{lr_1} won {agent_1_wins} times, Agent_{iter_2}_{lr_2} won {agent_2_wins} times.\n")

    # Save results to JSON file
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nAll simulation results saved to {output_file}")

if __name__ == "__main__":
    iterations = [50]
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1/np.sqrt(2), 0.8, 0.9]
    num_games = 100

    # Run the simulations
    run_simulations(iterations, learning_rates, num_games, output_file="backup_diff_lr.json")
