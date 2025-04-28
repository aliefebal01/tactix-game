
import os
import numpy as np
from scripts_1.train_masked_opp_trans import TrainAndPlot_masked_opp_trans

def main():    

    
    
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pretrained_model_path_30 = os.path.join(
        project_root,
        "training_results_1",
        "training_results_3x3_opp_trans",
        "models",
        "network_hl_128_128_128_gamma_0.70_bs_128_tufq_200_mcts_iter_15_mcts_lr_0.3_wr_68_tr_-25258.pth"
    )
    
    

    #small
    experiment_mcts2 = TrainAndPlot_masked_opp_trans(env_size=3, n_episodes=1000000, max_t=1000, batch_sizes=[128], layers= [[128,128, 128]],
                                                                    gammas=[0.7], epsilon_min=0.05, epsilon_max=0.05, epsilon_decay=0.9999, 
                                                                    memory_capacity=25000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    results_mcts = experiment_mcts2.run_experiments(pretrained_model_path=pretrained_model_path_30);


    # #medium
    # experiment_mcts2 = TrainAndPlot_basic(env_size=3, n_episodes=500000, max_t=1000, batch_sizes=[256], layers= [[256, 256, 256, 256, 256]],
    #                                                                 gammas=[0.75], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.99995, 
    #                                                                 memory_capacity=50000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=15, mcts_lr=0.3)
    # results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()