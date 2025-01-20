import numpy as np
from scripts.train_dqns_against import TrainAndPlot_dqns_against

def main():

    # on batch_sizes, gammas, memory_capacities, and layers: first element is for agent 1, second element is for agent 2



    experiment_dqns = TrainAndPlot_dqns_against(n_episodes=1000000, max_t=1000, batch_sizes=[128, 256], layers= [[128, 128, 128], [256, 256, 256, 256, 256]],
                                        gammas=[0.7, 0.7], epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.99995, 
                                        memory_capacities=[50000, 100000], device='cpu', target_update_freqs=[100, 100], lr=0.0001, log_interval=1000)
    print("tarinin medium and small dqns with attention against each other")
    results_dqns = experiment_dqns.run_experiments();

if __name__ == "__main__":
    main()