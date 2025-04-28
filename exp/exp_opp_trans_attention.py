
import os
import numpy as np
from scripts_1.train_opp_trans_attention import TrainAndPlot_opp_trans_attention

def main():    

    
    
    
    
    
    
    # #small
    # experiment_mcts2 = TrainAndPlot_opp_trans_attention(env_size=5, n_episodes=500000, max_t=1000, batch_sizes=[128], layers= [[128,128, 128]],
    #                                                                 gammas=[0.6, 0.55], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.99995, 
    #                                                                 memory_capacity=50000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    # results_mcts = experiment_mcts2.run_experiments();


    # #medium
    # experiment_mcts2 = TrainAndPlot_opp_trans_attention(env_size=6, n_episodes=300000, max_t=1000, batch_sizes=[1024], layers= [[256, 256, 256, 256, 256]],
    #                                                                 gammas=[0.65], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.9997, 
    #                                                                 memory_capacity=200000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    # results_mcts = experiment_mcts2.run_experiments();

        #large
    experiment_mcts2 = TrainAndPlot_opp_trans_attention(env_size=7, n_episodes=250000, max_t=1000, batch_sizes=[512], layers= [[512, 512, 512, 512, 512, 512, 512]],
                                                                    gammas=[0.65], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.999, 
                                                                    memory_capacity=200000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()