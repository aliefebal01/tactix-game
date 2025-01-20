import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.dqn_agent_attention import DQNAgent as DQNAgent_1
from scripts.dqn_agent_attention import DQNAgent as DQNAgent_2
from tactix.utils import *
from tactix.tactixEnvironment_without_opp import TactixEnvironment
from tactix.tactixGame import TactixGame
from tqdm import tqdm 
import os

LAYERS = [[128, 128, 128], [128, 128, 128]]
GAMMAS = [0.7, 0.75]

class TrainAndPlot_dqns_against:
    def __init__(self,
                env_size = 5,
                n_episodes=100000, 
                max_t=1000, 
                batch_sizes=[64, 75],
                layers = None, 
                gammas = None, 
                epsilon_min=0.01, 
                epsilon_max=1.0, 
                epsilon_decay=0.99995, 
                memory_capacities=[50000, 50000], 
                device='cpu', 
                target_update_freqs=[100, 100], 
                lr=1e-4,
                log_interval=1000):
        
        self.env = TactixEnvironment(board_size=env_size)
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.batch_sizes = batch_sizes
        self.layers = layers if layers else LAYERS
        self.gammas = gammas if gammas else GAMMAS
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.memory_capacities = memory_capacities
        self.device = device
        self.target_update_freqs = target_update_freqs
        self.lr = lr
        self.log_interval = log_interval
       
    def run_training(self, models_dir, pretrained_model_path_1=None, pretrained_model_path_2=None):
    
        # Create agent_1
        agent_1 = DQNAgent_1(
            state_size=self.env.game.height ** 2,
            action_size=self.env.game.height ** 3,
            layer_sizes=self.layers[0],
            lr=self.lr,
            gamma=self.gammas[0],
            epsilon_start=self.epsilon_max,
            epsilon_end=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            memory_capacity=self.memory_capacities[0],
            pretrained_model_path=pretrained_model_path_1
        )

        # Logging for agent_1
        rewards_log_1 = []
        cumulative_rewards_log_1 = []
        win_log_1 = []
        epsilon_log_1 = []
        total_reward_1 = 0.0  # For cumulative tracking for agent_1
        
        # Initialize variables for tracking the ultimate best model for agent_1
        ultimate_best_win_rate_1 = float('-inf')
        ultimate_best_cumulative_reward_1 = float('-inf')
        ultimate_best_model_path_1 = None  # Track the path of the ultimate best model for agent_1

        # Create agent_2
        agent_2 = DQNAgent_2(
            state_size=self.env.game.height ** 2,
            action_size=self.env.game.height ** 3,
            layer_sizes=self.layers[1],
            lr=self.lr,
            gamma=self.gammas[1],
            epsilon_start=self.epsilon_max,
            epsilon_end=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            memory_capacity=self.memory_capacities[1],
            pretrained_model_path=pretrained_model_path_2
        )

        # Logging for agent_2
        rewards_log_2 = []
        cumulative_rewards_log_2 = []
        win_log_2 = []
        epsilon_log_2 = []
        total_reward_2 = 0.0  # For cumulative tracking for agent_2
        
        # Initialize variables for tracking the ultimate best model for agent_2
        ultimate_best_win_rate_2 = float('-inf')
        ultimate_best_cumulative_reward_2 = float('-inf')
        ultimate_best_model_path_2 = None  # Track the path of the ultimate best model for agent_2

        #Training loop
        progress_bar = tqdm(range(self.n_episodes), desc="Initializing Training...", unit="episode", dynamic_ncols=True)
        for episode in progress_bar:
            state, valid_moves_mask = self.env.reset()

            episode_reward_1 = 0
            episode_reward_2 = 0
            done = False

            while not done:
            
                if self.env.game.current_player == 1:
                    self.env.state = self.env.game.getPieces()
                    state = self.env._get_observation()
                    valid_moves_mask = self.env._generate_valid_moves_mask()       
                    state = state.view(-1).unsqueeze(0)               # Shape: [1, state_size]
                    valid_moves_mask = valid_moves_mask.unsqueeze(0)  # Shape: [1, action_size]

                    action = agent_1.select_action(state, valid_moves_mask)
                    next_state, reward, done = self.env.step(action)  # next_state after agent made a move -> s'
                    next_state = next_state.view(-1).unsqueeze(0)     # shape: [1, state_size]

                    agent_1.memory.push((state.cpu(), 
                                    torch.tensor(action).cpu(), 
                                    reward, 
                                    next_state.cpu(),
                                    #next_state_valid_moves_mask.cpu(), 
                                    done))

                    # Train
                    agent_1.train_step(self.batch_sizes[0]) 
                    if reward == 1 or reward == -1:
                        episode_reward_1 += reward

                if not done:
                    self.env.state = self.env.game.getPieces()
                    state = self.env._get_observation()
                    valid_moves_mask = self.env._generate_valid_moves_mask()       
                    state = state.view(-1).unsqueeze(0)               # Shape: [1, state_size]
                    valid_moves_mask = valid_moves_mask.unsqueeze(0)  # Shape: [1, action_size]

                    action = agent_2.select_action(state, valid_moves_mask)
                    next_state, reward, done = self.env.step(action)  # next_state after agent made a move -> s'
                    next_state = next_state.view(-1).unsqueeze(0)     # shape: [1, state_size]

                    agent_2.memory.push((state.cpu(), 
                                    torch.tensor(action).cpu(), 
                                    reward, 
                                    next_state.cpu(),
                                    #next_state_valid_moves_mask.cpu(), 
                                    done))

                    # Train
                    agent_2.train_step(self.batch_sizes[1])
                    if reward == 1 or reward == -1:
                        episode_reward_2 += reward
            
            
            #Update targets
            if episode % self.target_update_freqs[0] == 0:
                agent_1.update_target_network()
            if episode % self.target_update_freqs[1] == 0:
                agent_2.update_target_network()
                    
            
            if episode_reward_1 > episode_reward_2:
                episode_reward_1 = 1
                episode_reward_2 = 0
            else:
                episode_reward_2 = 1
                episode_reward_1 = 0
            
            
            # Logging 1
            total_reward_1 += episode_reward_1
            rewards_log_1.append(episode_reward_1)
            cumulative_rewards_log_1.append(total_reward_1)
            win_log_1.append(1 if episode_reward_1 > 0 else 0)
            epsilon_log_1.append(agent_1.epsilon)

            # Logging 2
            total_reward_2 += episode_reward_2
            rewards_log_2.append(episode_reward_2)
            cumulative_rewards_log_2.append(total_reward_2)
            win_log_2.append(1 if episode_reward_2 > 0 else 0)
            epsilon_log_2.append(agent_2.epsilon)
            

            if len(win_log_1) >= 200:
                avg_win_rate_1 = 100.0 * np.mean(win_log_1[-200:])
                current_cumulative_reward_1 = cumulative_rewards_log_1[-1] if cumulative_rewards_log_1 else 0

                # Save the model only if:
                # 1. The new win rate is greater than the best win rate seen so far, or
                # 2. The new win rate equals the best win rate, but the cumulative reward is higher
                if (avg_win_rate_1 > ultimate_best_win_rate_1) or (
                        avg_win_rate_1 == ultimate_best_win_rate_1 and current_cumulative_reward_1 > ultimate_best_cumulative_reward_1):
                    ultimate_best_win_rate_1 = avg_win_rate_1
                    ultimate_best_cumulative_reward_1 = current_cumulative_reward_1

                    # Update the model state and name
                    ultimate_best_model_state_1 = agent_1.q_network.state_dict()
                    ultimate_best_model_name_1 = (
                        f"network_hl_{'_'.join(map(str, self.layers[0]))}_gamma_{self.gammas[0]:.2f}_"
                        f"bs_{self.batch_sizes[0]}_tufq_{self.target_update_freqs[0]}_"
                        f"wr_{int(ultimate_best_win_rate_1)}_tr_{int(ultimate_best_cumulative_reward_1)}.pth"
                    )

            
            if len(win_log_2) >= 200:
                avg_win_rate_2 = 100.0 * np.mean(win_log_2[-200:])
                current_cumulative_reward_2 = cumulative_rewards_log_2[-1] if cumulative_rewards_log_2 else 0

                # Save the model only if:
                # 1. The new win rate is greater than the best win rate seen so far, or
                # 2. The new win rate equals the best win rate, but the cumulative reward is higher
                if (avg_win_rate_2 > ultimate_best_win_rate_2) or (
                        avg_win_rate_2 == ultimate_best_win_rate_2 and current_cumulative_reward_2 > ultimate_best_cumulative_reward_2):
                    ultimate_best_win_rate_2 = avg_win_rate_2
                    ultimate_best_cumulative_reward_2 = current_cumulative_reward_2

                    # Update the model state and name
                    ultimate_best_model_state_2 = agent_2.q_network.state_dict()
                    ultimate_best_model_name_2 = (
                        f"network_hl_{'_'.join(map(str, self.layers[1]))}_gamma_{self.gammas[1]:.2f}_"
                        f"bs_{self.batch_sizes[1]}_tufq_{self.target_update_freqs[1]}_"
                        f"wr_{int(ultimate_best_win_rate_2)}_tr_{int(ultimate_best_cumulative_reward_2)}.pth"
                    )
                
            if episode % 10000 == 0 and len(win_log_1) >= 200:
                    avg_reward_1 = np.mean(rewards_log_1[-200:])
                    win_rate_1 = 100.0 * np.mean(win_log_1[-200:])
                    avg_reward_2 = np.mean(rewards_log_2[-200:])
                    win_rate_2 = 100.0 * np.mean(win_log_2[-200:])
                    progress_bar.set_description(
                        f"Agent 1: R={avg_reward_1:.2f}, W={win_rate_1:.2f}%, E={agent_1.epsilon:.3f} | "
                        f"Agent 2: R={avg_reward_2:.2f}, W={win_rate_2:.2f}%, E={agent_2.epsilon:.3f}"
                    )

        if ultimate_best_model_state_1 and ultimate_best_model_name_1:
            ultimate_best_model_path_1 = os.path.join(models_dir, ultimate_best_model_name_1)
            torch.save(ultimate_best_model_state_1, ultimate_best_model_path_1)
            print(f"Agent 1's ultimate best model saved: {ultimate_best_model_path_1}")

        if ultimate_best_model_state_2 and ultimate_best_model_name_2:
            ultimate_best_model_path_2 = os.path.join(models_dir, ultimate_best_model_name_2)
            torch.save(ultimate_best_model_state_2, ultimate_best_model_path_2)
            print(f"Agent 2's ultimate best model saved: {ultimate_best_model_path_2}")

        return rewards_log_1, cumulative_rewards_log_1, win_log_1, epsilon_log_1, ultimate_best_win_rate_1, ultimate_best_cumulative_reward_1, rewards_log_2, cumulative_rewards_log_2, win_log_2, epsilon_log_2, ultimate_best_win_rate_2, ultimate_best_cumulative_reward_2



    def run_experiments(self, pretrained_model_path_1=None, pretrained_model_path_2=None):
    
         # Centralized directory setup
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(project_root, "training_results")
        save_dir = os.path.join(base_dir,f"training_results_{self.env.game.height}x{self.env.game.height}_dqns_against_each_other")
        models_dir = os.path.join(save_dir, "models")
        plots_dir = os.path.join(save_dir, "plots")

        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        results_1 = {}
        results_2 = {}

        print(f"Training: \n"
              f"Agent_1: Layers={self.layers[0]}, Gamma={self.gammas[0]}, Batch Size={self.batch_sizes[0]}, mem_cap={self.memory_capacities[0]}, Target Update={self.target_update_freqs[0]}"
              f"Agent_2: Layers={self.layers[1]}, Gamma={self.gammas[1]}, Batch Size={self.batch_sizes[1]}, mem_cap={self.memory_capacities[1]}, Target Update={self.target_update_freqs[1]}"
              )    

        r_log_1, c_log_1, w_log_1, e_log_1, ultimate_best_win_rate_1, ultimate_best_cumulative_reward_1, r_log_2, c_log_2, w_log_2, e_log_2, ultimate_best_win_rate_2, ultimate_best_cumulative_reward_2 = self.run_training(models_dir, pretrained_model_path_1=pretrained_model_path_1, pretrained_model_path_2=pretrained_model_path_2)

        results_1[(tuple(self.layers[0]), self.gammas[0])] = (r_log_1, c_log_1, w_log_1, e_log_1)
        results_2[(tuple(self.layers[1]), self.gammas[1])] = (r_log_2, c_log_2, w_log_2, e_log_2)

        # Plot results for this combination
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        
        # Prepare parameter text
        parameter_text = (
            f"n_episodes={self.n_episodes}, max_t={self.max_t}, batch_sizes={self.batch_sizes[0]}, {self.batch_sizes[1]},\n"
            f"board_size = {self.env.game.height}x{self.env.game.height}, layers={self.layers[0]}, {self.layers[1]}, gammas={self.gammas[0]:.2f}, {self.gammas[1]:.2f}\n"
            f"epsilon_min={self.epsilon_min}, epsilon_max={self.epsilon_max}, epsilon_decay={self.epsilon_decay},\n"
            f"memory_capacities={self.memory_capacities[0]}, {self.memory_capacities[1]}, device={self.device}, target_update_freqs={self.target_update_freqs[0]}, {self.target_update_freqs[1]},\n"
            f"lr={self.lr}"
        )

        # 1) Rewards
        axs[0].plot(r_log_1, label="Agent 1 Rewards", color='blue')
        axs[0].plot(r_log_2, label="Agent 2 Rewards", color='orange')
        rolling_avg_r_1 = [np.mean(r_log_1[max(0, i - 500):i + 1]) for i in range(len(r_log_1))]
        rolling_avg_r_2 = [np.mean(r_log_2[max(0, i - 500):i + 1]) for i in range(len(r_log_2))]
        axs[0].plot(rolling_avg_r_1, label="Agent 1 Avg Rewards (Last 500)", linestyle='--', color='blue')
        axs[0].plot(rolling_avg_r_2, label="Agent 2 Avg Rewards (Last 500)", linestyle='--', color='orange')
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].set_title("Rewards", fontsize=14)
        axs[0].legend()
        axs[0].grid()

        # 2) Cumulative Rewards
        axs[1].plot(c_log_1, label="Agent 1 Cumulative Rewards", color='blue')
        axs[1].plot(c_log_2, label="Agent 2 Cumulative Rewards", color='orange')
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Cumulative Reward")
        axs[1].set_title("Cumulative Rewards", fontsize=14)
        axs[1].legend()
        axs[1].grid()

        # 3) Win Rate
        rolling_win_1 = [100.0 * np.mean(w_log_1[max(0, i - 500):i + 1]) for i in range(len(w_log_1))]
        rolling_win_2 = [100.0 * np.mean(w_log_2[max(0, i - 500):i + 1]) for i in range(len(w_log_2))]
        axs[2].plot(rolling_win_1, label="Agent 1 Win Rate (Last 500)", color='blue')
        axs[2].plot(rolling_win_2, label="Agent 2 Win Rate (Last 500)", color='orange')
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Win Rate (%)")
        axs[2].set_title("Win Rate", fontsize=14)
        axs[2].legend()
        axs[2].grid()

        # Add parameter text at the top
        fig.text(
            0.5, 1.02,  # Position above the subplots
            parameter_text,
            ha='center',
            va='bottom',
            fontsize=9
        )

        # Convert the parameter_text into a single-line string for the file name
        parameters_for_filename = (
            f"numep_{self.n_episodes}_bs_{self.batch_sizes[0]}_{self.batch_sizes[1]}_"
            f"hl_{'_'.join(map(str, self.layers[0]))}_{'_'.join(map(str, self.layers[1]))}_"
            f"gamma_{self.gammas[0]:.2f}_{self.gammas[1]:.2f}_"
            f"mem_cap_{self.memory_capacities[0]}_{self.memory_capacities[1]}_"
            f"tufq_{self.target_update_freqs[0]}_{self.target_update_freqs[1]}_lr_{self.lr}"
        )

        # Replace invalid characters for filenames
        parameters_for_filename = parameters_for_filename.replace(":", "_").replace(" ", "_").replace("/", "_").replace(".", "_")

        # Adjust layout to leave space at the top for the parameter text
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave 15% space at the top
        plt.subplots_adjust(top=0.8)  # Adjust top space explicitly

        # Save the plot
        plot_name = f"{parameters_for_filename}.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path, bbox_inches="tight")  # Save all elements, ensuring no clipping
        plt.close()

        return results_1, results_2