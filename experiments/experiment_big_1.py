import numpy as np
from scripts.train_without_opp_move_mh_attention_mcts2 import TrainAndPlot_without_opp_move_mh_attention_mcts2

def main():



    experiment_mcts = TrainAndPlot_without_opp_move_mh_attention_mcts2(n_episodes=250000, max_t=1000, batch_sizes=[512], layers= [[512, 512, 512, 512, 512, 512, 512]],
                                                                    gammas=[0.7], epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.9995, 
                                                                    memory_capacity=200000, device='cpu', target_update_freq=100, lr=0.0001, log_interval=1000, mcts_iteration=50, mcts_lr=0.4)
    print("big training for 250k episodes with attention mh_size = 3")
    results_mcts = experiment_mcts.run_experiments();

if __name__ == "__main__":
    main()