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
        self.multihead_attention = nn.MultiheadAttention(embed_dim=state_size, num_heads=1, batch_first=True)
        
        
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
        # Add a sequence dimension: (batch_size, state_size) -> (batch_size, 1, state_size)
        #x = x.unsqueeze(1)

        # Input shape: (batch_size, state_size)
        attn_output, _ = self.multihead_attention(x, x, x)  # Self-attention

        # Remove the sequence dimension: (batch_size, 1, state_size) -> (batch_size, state_size)
        x = attn_output
        
        # Pass through the rest of the network
        return self.network(x)  # Outputs Q-values for each action