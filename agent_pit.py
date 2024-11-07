from tactix.tactixGame import TactixGame
from agents.mcts.mcts_agent import MCTSAgent
from tactix.tactixMove import Move
from agents.mcts.mcts_node import MCTSNode

def agents_play():
    # Initialize game and agents
    game = TactixGame()
    agent_10000 = MCTSAgent(player=1, iterations=10000)
    agent_30000 = MCTSAgent(player=-1, iterations=30000)
    current_node = MCTSNode(game)

    print("Game started!")

    while current_node.state.getGameEnded() is None:

        if current_node.state.current_player == agent_10000.player:
            # Agent 10000's turn
            print("\n Agent_10000 is making a move...")
            best_node = agent_10000.best_child(current_node)
            current_node = best_node
            current_node.state.display()

        else:
            # Agent 30000's turn
            print("\n Agent 30000 is making a move...")
            best_node = agent_30000.best_child(current_node)
            current_node = best_node
            current_node.state.display()

    # Announce winner 
    winner = current_node.state.getGameEnded().winner
    if winner == agent_10000.player:
        print("Agent_10000 wins!")
    else:
        print("Agent_30000 wins!")

if __name__ == "__main__":
    agents_play()
            