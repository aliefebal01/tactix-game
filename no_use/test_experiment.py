
import numpy as np
from scripts.train_without_opp_move_attention_random import TrainAndPlot_without_opp_move_attention_random

def main():



    experiment_mcts = TrainAndPlot_without_opp_move_attention_random(n_episodes=1000, max_t=1000, batch_sizes=[128], layers= [[128,128,128]],
                                                                    gammas=[0.7], epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.9995, 
                                                                    memory_capacity=50000, device='cpu', target_update_freq=100, lr=0.0001, log_interval=1000)
    results_mcts = experiment_mcts.run_experiments();

if __name__ == "__main__":
    main()