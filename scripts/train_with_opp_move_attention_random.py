import torch
from scripts.dqn_agent_attention import DQNAgent
import numpy as np
from tactix.utils import *
from tactix.tactixEnvironment_with_opp_random import TactixEnvironment
from tactix.tactixGame import TactixGame
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os

LAYERS = [[50, 125], [125, 250, 250, 125]]
GAMMAS = [0.95, 0.9, 0.85]



class TrainAndPlot_with_opp_move_attention_random:
    def __init__(self,
                env_size = 5,
                n_episodes=100000, 
                max_t=1000, 
                batch_sizes=[64],
                layers = None, 
                gammas = None, 
                epsilon_min=0.05, 
                epsilon_max=1.0, 
                epsilon_decay=0.99995, 
                memory_capacity=50000, 
                device='cpu', 
                target_update_freq=1000, 
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
        self.memory_capacity = memory_capacity
        self.device = device
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.log_interval = log_interval

    def run_training(self, layer_structure, gamma, models_dir, batch_size):
    


        # Create agent
        agent = DQNAgent(
            state_size=self.env.game.height ** 2,
            action_size=self.env.game.height ** 3,
            layer_sizes=layer_structure,
            lr=self.lr,
            gamma=gamma,
            epsilon_start=self.epsilon_max,
            epsilon_end=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            memory_capacity=self.memory_capacity
        )
        
        # Logging
        rewards_log = []
        cumulative_rewards_log = []
        win_log = []
        epsilon_log = []
        total_reward = 0.0  # For cumulative tracking
        

        # Initialize variables for tracking the ultimate best model
        ultimate_best_win_rate = float('-inf')
        ultimate_best_cumulative_reward = float('-inf')
        ultimate_best_model_path = None  # Track the path of the ultimate best model
        
        progress_bar = tqdm(range(self.n_episodes), desc="Initializing Training...", unit="episode", dynamic_ncols=True)
        for episode in progress_bar:
            state, valid_moves_mask = self.env.reset()
            state = state.view(-1).unsqueeze(0)         # shape: [1, state_size]
            valid_moves_mask = valid_moves_mask.unsqueeze(0)  # shape: [1, action_size]

            episode_reward = 0
            done = False
            
            
            while not done:
                action = agent.select_action(state, valid_moves_mask)
                next_state, reward, done, valid_moves_mask = self.env.step(action)
                next_state = next_state.view(-1).unsqueeze(0)
                valid_moves_mask = valid_moves_mask.unsqueeze(0) # shape: [1, action_size]
                #next_state_valid_moves_mask = valid_moves_mask  # shape: [1, action_size]
                
                # Push to replay
                agent.memory.push((state.cpu(), 
                                torch.tensor(action).cpu(), 
                                reward, 
                                next_state.cpu(),
                                #next_state_valid_moves_mask.cpu(), 
                                done))

                # Train
                agent.train_step(batch_size)

                state = next_state
                episode_reward += reward

            # Update target
            if episode % self.target_update_freq == 0:
                agent.update_target_network()

            # Logging
            total_reward += episode_reward
            rewards_log.append(episode_reward)
            cumulative_rewards_log.append(total_reward)
            win_log.append(1 if episode_reward > 0 else 0)
            epsilon_log.append(agent.epsilon)


            if len(win_log) >= 200:
                avg_win_rate = 100.0 * np.mean(win_log[-200:])
                current_cumulative_reward = cumulative_rewards_log[-1] if cumulative_rewards_log else 0

                # Save the model only if:
                # 1. The new win rate is greater than the best win rate seen so far, or
                # 2. The new win rate equals the best win rate, but the cumulative reward is higher
                if (avg_win_rate > ultimate_best_win_rate) or (
                        avg_win_rate == ultimate_best_win_rate and current_cumulative_reward > ultimate_best_cumulative_reward):
                    ultimate_best_win_rate = avg_win_rate
                    ultimate_best_cumulative_reward = current_cumulative_reward

                    # Update the model state and name
                    ultimate_best_model_state = agent.q_network.state_dict()
                    ultimate_best_model_name = (
                        f"network_hl_{'_'.join(map(str, layer_structure))}_gamma_{gamma:.2f}_"
                        f"bs_{batch_size}_tufq_{self.target_update_freq}_"
                        f"wr_{int(ultimate_best_win_rate)}_tr_{int(ultimate_best_cumulative_reward)}.pth"
                    )


            # Print progress occasionally
            # if (episode+1) % self.log_interval == 0:  # Log interval
            #     avg_reward = np.mean(rewards_log[-100:]) if len(rewards_log) > 100 else np.mean(rewards_log)
            #     win_rate = 100.0 * np.mean(win_log[-100:]) if len(win_log) > 100 else 100.0 * np.mean(win_log)
            #     print(f"[{episode+1}/{self.n_episodes}] Layers={layer_structure}, Gamma={gamma}, "
            #         f"AvgReward(Last100)={avg_reward:.2f}, WinRate(Last100)={win_rate:.2f}%, Eps={agent.epsilon:.3f}")

            if episode % 10000 == 0 and len(win_log) >= 200:
                avg_reward = np.mean(rewards_log[-200:])
                win_rate = 100.0 * np.mean(win_log[-200:])
                progress_bar.set_description(
                    f"AvgReward={avg_reward:.2f}, WinRate={win_rate:.2f}%, "
                    f"Eps={agent.epsilon:.3f}"
                )
                
        if ultimate_best_model_state and ultimate_best_model_name:
            ultimate_best_model_path = os.path.join(models_dir, ultimate_best_model_name)
            torch.save(ultimate_best_model_state, ultimate_best_model_path)
            print(f"Ultimate best model saved: {ultimate_best_model_path}")

        return rewards_log, cumulative_rewards_log, win_log, epsilon_log, ultimate_best_win_rate, ultimate_best_cumulative_reward





    def run_experiments(self):
        
        # Centralized directory setup
        base_dir = "/Users/alibal/Desktop/tactix_training"  # Adjust to your system's directory path
        save_dir = os.path.join(base_dir, f"training_results_{self.env.game.height}x{self.env.game.height}_randomopponent_s'_after_opponent_with_attention")
        models_dir = os.path.join(save_dir, "models")
        plots_dir = os.path.join(save_dir, "plots")

        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        results = {}  # (layer_tuple, gamma) -> (rewards_log, cumulative_rewards_log, win_log, epsilon_log)
        for batch_size in self.batch_sizes:
            for layer in self.layers:
                for gamma in self.gammas:
                    print(f"=== Training with LayerStructure={layer}, Gamma={gamma}, Batch Size={batch_size}, Epsilon(max, min)={self.epsilon_max, self.epsilon_min}, mem_cap={self.memory_capacity}, Target Update={self.target_update_freq} ===")
                    r_log, c_log, w_log, e_log, ultimate_best_win_rate, ultimate_best_cumulative_reward = self.run_training(layer, gamma, models_dir, batch_size)
                    results[(tuple(layer), gamma)] = (r_log, c_log, w_log, e_log)
                    
                    # Plot results for this combination
                    fig, axs = plt.subplots(3, 1, figsize=(16, 16))
                    
                    # Prepare parameter text
                    parameter_text = (
                        f"n_episodes={self.n_episodes}, max_t={self.max_t}, batch_size={batch_size},\n"
                        f"board_size = {self.env.game.height}x{self.env.game.height}, layers={layer}, gamma={gamma:.2f},\n"
                        f"epsilon_min={self.epsilon_min}, epsilon_max={self.epsilon_max}, epsilon_decay={self.epsilon_decay},\n"
                        f"memory_capacity={self.memory_capacity}, device={self.device}, target_update_freq={self.target_update_freq},\n"
                        f"lr={self.lr}, log_interval={self.log_interval}"
                    )
                    
                    # 1) Rewards
                    axs[0].plot(r_log, label="Rewards")
                    rolling_avg_r = [np.mean(r_log[max(0, i-1000):i+1]) for i in range(len(r_log))]
                    axs[0].plot(rolling_avg_r, label="Average Rewards (Last 1000)")
                    axs[0].set_xlabel("Episode")
                    axs[0].set_ylabel("Reward")
                    axs[0].set_title(f"Rewards - Layers={layer}, Gamma={gamma}", fontsize=14)
                    
                    axs[0].legend()
                    axs[0].grid()
                    
                    # 2) Cumulative Rewards
                    axs[1].plot(c_log, label="Cumulative Rewards")
                    axs[1].set_xlabel("Episode")
                    axs[1].set_ylabel("Total Reward")
                    axs[1].set_title(f"Cumulative Rewards - Layers={layer}, Gamma={gamma}")
                    axs[1].legend()
                    axs[1].grid()
                    
                    # 3) Win Rate
                    rolling_win = [100.0*np.mean(w_log[max(0, i-1000):i+1]) for i in range(len(w_log))]
                    axs[2].plot(rolling_win, label="Win Rate (Last 1000 Episodes)")
                    axs[2].set_xlabel("Episode")
                    axs[2].set_ylabel("Win Rate (%)")
                    axs[2].set_title(f"Win Rate - Layers={layer}, Gamma={gamma}")
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
                        f"numep_{self.n_episodes}_bs_{batch_size}_"
                        f"hl_{'_'.join(map(str, layer))}_"
                        f"gamma_{gamma:.2f}_"
                        f"mem_cap_{self.memory_capacity}_"
                        f"tufq_{self.target_update_freq}_lr_{self.lr}_"
                        f"wr_{int(ultimate_best_win_rate)}_tr_{int(ultimate_best_cumulative_reward)}"
                    )

                    # Replace any characters that are invalid in file names (e.g., colons, slashes, spaces)
                    parameters_for_filename = parameters_for_filename.replace(":", "_").replace(" ", "_").replace("/", "_").replace(".", "_")

                    # Adjust layout to leave space at the top for the parameter text
                    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Leave 5% space at the top
                    plt.subplots_adjust(top=0.8)  # Adjust top space explicitly

                    # Save the plot
                    plot_name = f"{parameters_for_filename}.png"
                    plot_path = os.path.join(plots_dir, plot_name)
                    plt.savefig(plot_path, bbox_inches="tight")  # Save all elements, ensuring no clipping
                    plt.show()
                    
        return results