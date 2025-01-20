
import os
import numpy as np
from scripts.train_without_opp_move_attention_in_re_mcts2 import TrainAndPlot_without_opp_move_attention_in_re_mcts2

def main():

    

    experiment_mcts2 = TrainAndPlot_without_opp_move_attention_in_re_mcts2(n_episodes=250000, max_t=1000, batch_sizes=[256], layers= [[256,256,256, 256, 256]],
                                                                    gammas=[0.7], epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.9995, 
                                                                    memory_capacity=100000, device='cpu', target_update_freq=100, lr=0.0001, log_interval=100, mcts_iteration=50, mcts_lr=0.4)
    print("training medium with attention and intermediate rewards for 250k episodes")
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()
    