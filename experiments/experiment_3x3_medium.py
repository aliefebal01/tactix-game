from scripts.train_without_opp_move_attention_mcts2 import TrainAndPlot_without_opp_move_attention_mcts2

def main():



    experiment_mcts = TrainAndPlot_without_opp_move_attention_mcts2(env_size=3, n_episodes=1000000, max_t=1000, batch_sizes=[256], layers= [[256, 256, 256, 256, 256]],
                                                                    gammas=[0.7], epsilon_min=0.05, epsilon_max=1.0, epsilon_decay=0.999987, 
                                                                    memory_capacity=50000, device='cpu', target_update_freq=200, lr=0.0001, log_interval=1000, mcts_iteration=10, mcts_lr=0.4)
    results_mcts = experiment_mcts.run_experiments();

if __name__ == "__main__":
    main()