import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    A DQN with a single attention layer after the input state.
    """
    def __init__(self, state_size, action_size, layer_sizes):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Attention layer
        self.attention_layer = nn.Linear(state_size, state_size)
        self.attention_activation = nn.Softmax(dim=-1)  # Softmax to generate attention weights
        
        # Hidden layers
        layers = []
        input_dim = state_size
        for hidden_size in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Final layer: from the last hidden layer to the action output
        layers.append(nn.Linear(input_dim, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the DQN with an attention layer.
        x: Input state tensor, shape [batch_size, state_size].
        """
        # Attention mechanism
        attention_weights = self.attention_activation(self.attention_layer(x))  # [batch_size, state_size]
        x = x * attention_weights  # Apply attention weights to the input
        
        # Pass through the rest of the network
        return self.network(x)  # Outputs Q-values for each action