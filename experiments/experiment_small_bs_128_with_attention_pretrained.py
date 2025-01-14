
import numpy as np
import os
from scripts.train_without_opp_move_attention_mcts2 import TrainAndPlot_without_opp_move_attention_mcts2

def main():

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pretrained_model_path_30 = os.path.join(
        project_root,
        "training_results",
        "training_results_5x5_with_attention_mcts2",
        "models",
        "network_hl_128_128_128_gamma_0.70_bs_128_tufq_100_mcts_iter_100_mcts_lr_0.4_wr_30_tr_-75425.pth"
    )

    experiment_mcts_pre_30 = TrainAndPlot_without_opp_move_attention_mcts2(n_episodes=100000, max_t=1000, batch_sizes=[128], layers= [[128,128,128]],
                                                                    gammas=[0.7], epsilon_min=0.01, epsilon_max=0.7, epsilon_decay=0.9995, 
                                                                    memory_capacity=50000, device='cpu', target_update_freq=100, lr=0.0001, log_interval=1000, mcts_iteration=100, mcts_lr=0.4)
    results_mcts = experiment_mcts_pre_30.run_experiments(pretrained_model_path=pretrained_model_path_30);

if __name__ == "__main__":
    main()
    