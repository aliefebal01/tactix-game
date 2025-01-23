import os
import numpy as np
from scripts.train_end_game_attention import TrainAndPlot_end_game_attention

def main():

    
    
    
    
    
    


    experiment_mcts2 = TrainAndPlot_end_game_attention(n_episodes=300000, max_t=1000, batch_sizes=[256], layers= [[256, 256, 256, 256, 256]],
                                                                    gammas=[0.7], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.9999, 
                                                                    memory_capacity=50000, device='cpu', target_update_freq=100, lr=0.0001, log_interval=100, mcts_iteration=10, mcts_lr=0.4)
    
    print("training medium model end game last 8 pieces low mcts iteration")
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()