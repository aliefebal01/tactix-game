import torch
import torch.nn as nn
import torch.optim as optim
from scripts_1.replay_memory import ReplayMemory
import random 
from scripts_1.dqn_attention import DQN

random.seed(42)

class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        layer_sizes,
        lr=1e-3, 
        gamma=0.9, 
        epsilon_start=1.0, 
        epsilon_end=0.01, 
        epsilon_decay=0.999876,
        memory_capacity=10000,
        device='cpu',
        pretrained_model_path=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        torch.manual_seed(42)  # Ensures deterministic weight initialization
        # Q-Networks (main + target)
        self.q_network = DQN(state_size, action_size, layer_sizes).to(self.device)
        self.target_network = DQN(state_size, action_size, layer_sizes).to(self.device)

        if pretrained_model_path:
            self.q_network.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_model_path}")

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Memory
        self.memory = ReplayMemory(capacity=memory_capacity)

    def select_action(self, state, valid_moves_mask):
        """
        Epsilon-greedy action selection with invalid move masking.
        state: shape (1, state_size)
        valid_moves_mask: shape (1, action_size) -> 1/0 for valid moves
        """
        if random.random() < self.epsilon:
            valid_indices = torch.where(valid_moves_mask[0] == 1)[0]
            action = random.choice(valid_indices.tolist())
            return action
        else:
            with torch.no_grad():
                q_values = self.q_network(state.to(self.device))  # (1, action_size)
                # Mask invalid actions by setting them to -inf
                q_values[valid_moves_mask == 0] = -float('inf')
                return q_values.argmax(dim=1).item()

    def update_target_network(self):
        """Update the target network to match the Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
 
    def train_step(self, batch_size):
        """Train the Q-network using one batch from experience replay."""
        if len(self.memory) < batch_size:
            return  # Not enough samples to train
        
        # Sample a batch of transitions
        transitions = self.memory.sample(batch_size)
        # transitions is a list of tuples: (state, action, reward, next_state, done)
        batch = list(zip(*transitions))

        states = torch.stack(batch[0]).to(self.device)          # shape: [batch_size, 1, state_size]
        actions = torch.stack(batch[1]).to(self.device)         # shape: [batch_size]
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(self.device)  # [batch_size]
        next_states = torch.stack(batch[3]).to(self.device)     # shape: [batch_size, 1, state_size]
        next_states_valid_moves_mask = torch.stack(batch[4]).to(self.device)  # shape: [batch_size, 1, action_size]
        dones = torch.tensor(batch[5], dtype=torch.bool).to(self.device)       # [batch_size]
        
        # Flatten states: we have [batch_size, 1, state_size] => [batch_size, state_size]
        states = states.view(states.size(0), -1)
        next_states = next_states.view(next_states.size(0), -1)

        # Flatten next_states_valid_moves_mask: [batch_size, 1, action_size] => [batch_size, action_size]
        # next_states_valid_moves_mask = next_states_valid_moves_mask.view(next_states_valid_moves_mask.size(0), -1)

        # Current Q-values
        q_values = self.q_network(states)
        # Gather Q-values for the taken actions
        # q_values shape is [batch_size, action_size], actions is [batch_size]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():  
            temp_next_q_values = self.target_network(next_states)
            temp_next_q_values[next_states_valid_moves_mask == 0] = float('inf')
            max_next_q_values = temp_next_q_values.min(1)[0]
            # max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * max_next_q_values
        
        # Loss and optimization
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)