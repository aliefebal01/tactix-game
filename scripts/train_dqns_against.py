import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.dqn_agent_dr_re_attention import DQNAgent as DQNAgent_1
from scripts.dqn_agent_dr_re import DQNAgent as DQNAgent_2
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
                target_update_freq=1000, 
                lr=1e-4,
                log_interval=1000:
        
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
        self.target_update_freq = target_update_freq
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

            state = state.view(-1).unsqueeze(0)               # shape: [1, state_size]
            valid_moves_mask = valid_moves_mask.unsqueeze(0)  # shape: [1, action_size]

            episode_reward_1 = 0
            episode_reward_2 = 0
            done = False

            while not done:

                if self.env.game.current_player == 1:
                                


