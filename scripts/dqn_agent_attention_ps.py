import torch
import torch.nn as nn
import torch.optim as optim
from scripts.replay_memory_ps import PrioritizedReplayMemory
import random
from scripts.dqn_attention import DQN

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
        alpha=0.6,  # Priority sampling exponent
        beta_start=0.4,  # Initial importance-sampling correction factor
        beta_end=1.0,  # Final importance-sampling correction factor
        beta_anneal_steps=100000,  # Steps to anneal beta
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
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.steps_done = 0  # Counter for annealing beta
        
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
        self.memory = PrioritizedReplayMemory(capacity=memory_capacity, alpha=alpha)

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
        """Train the Q-network using one batch from prioritized replay memory."""
        if len(self.memory) < batch_size:
            return  # Not enough samples to train

        self.steps_done += 1

        # Anneal beta for importance sampling correction
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.steps_done / self.beta_anneal_steps)
        
        # Sample a batch of transitions with priorities
        transitions, indices, weights = self.memory.sample(batch_size, beta)

        # Unpack transitions
        batch = list(zip(*transitions))
        states = torch.stack(batch[0]).to(self.device)
        actions = torch.tensor(batch[1], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_states = torch.stack(batch[3]).to(self.device)
        dones = torch.tensor(batch[4], dtype=torch.bool).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Flatten states
        states = states.view(states.size(0), -1)
        next_states = next_states.view(next_states.size(0), -1)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        # TD errors
        td_errors = q_values - target_q_values

        # Loss with importance-sampling correction
        loss = (weights * td_errors.pow(2)).mean()

        # Update priorities in memory
        self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)