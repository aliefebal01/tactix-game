
import os
import numpy as np
from scripts_1.train_opp_trans import TrainAndPlot_opp_trans

def main():    

    
    
    
    
    
    


    # experiment_mcts2 = TrainAndPlot_opp_trans(env_size=5, n_episodes=1000000, max_t=1000, batch_sizes=[128], layers= [[128,128, 128]],
    #                                                                 gammas=[0.7], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.99995, 
    #                                                                 memory_capacity=50000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    # results_mcts = experiment_mcts2.run_experiments();

    #medium
    experiment_mcts2 = TrainAndPlot_opp_trans(env_size=5, n_episodes=500000, max_t=1000, batch_sizes=[512], layers= [[256, 256, 256, 256, 256]],
                                                                    gammas=[0.65], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.99995, 
                                                                    memory_capacity=100000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()