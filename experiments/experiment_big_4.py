
import os
import numpy as np
from scripts.train_end_game_mh import TrainAndPlot_without_opp_move_mh_attention

def main():

    
    
    
    
    
    


    experiment_mcts2 = TrainAndPlot_without_opp_move_mh_attention(n_episodes=100000, max_t=1000, batch_sizes=[2048], layers= [[512, 512, 512, 512, 512 , 512, 512]],
                                                                    gammas=[0.7], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.99998, 
                                                                    memory_capacity=100000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=100, mcts_iteration=10, mcts_lr=0.4)
    print("training big model end game last 10 pieces low mcts iteration 10 and big batch size")
    results_mcts = experiment_mcts2.run_experiments();


if __name__ == "__main__":
    main()
    