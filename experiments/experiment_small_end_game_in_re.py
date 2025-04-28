import os
import numpy as np
from scripts.train_end_game_in_re import TrainAndPlot_end_game_in_re

def main():

    
    
    
    
    
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # pretrained_model_path_30 = os.path.join(
    #     project_root,
    #     "training_results",
    #     "training_results_5x5_with_attention_in_re_mcts2",
    #     "models",
    #     "network_hl_128_128_128_gamma_0.70_bs_128_tufq_200_mcts_iter_50_mcts_lr_0.4_wr_32_tr_-617889.pth"
    # )


    experiment_mcts2 = TrainAndPlot_end_game_in_re(n_episodes=1000000, max_t=1000, batch_sizes=[128], layers= [[128,128,128]],
                                                                    gammas=[0.7], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.999987, 
                                                                    memory_capacity=25000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=5, mcts_lr=0.4)
    print("training small model end_game int. rew. 10 pieces with low mcts iterations")
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()
    