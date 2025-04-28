
import os
import numpy as np
from scripts.train_with_mcts_trans_attention import TrainAndPlot_with_mcts_trans_attention

def main():

    
    
    
    
    
    


    experiment_mcts2 = TrainAndPlot_with_mcts_trans_attention(n_episodes=500000, max_t=1000, batch_sizes=[256], layers= [[256, 256, 256, 256, 256]],
                                                                    gammas=[0.75], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.9999, 
                                                                    memory_capacity=100000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=50, mcts_lr=0.4)
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()