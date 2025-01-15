import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_ATTENTION = True

class DQNWithDropoutAndResidual(nn.Module):
    def __init__(self, state_size, action_size, layer_sizes):
        super(DQNWithDropoutAndResidual, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_hidden_layers = len(layer_sizes)
        self.embedding_dim = layer_sizes[0]

        # Projection layer 
        self.projection =  nn.Linear(state_size, self.embedding_dim)

        # Attention layer
        if HAS_ATTENTION:
            # Attention - Single or Multihead
            num_heads = 4           # has to divide hidden_dim
            self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, batch_first=True)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for i in range(self.num_hidden_layers)])

        # Dropout probability
        dropout_prob = 0.5

        # ReLu and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
        # Output layer
        self.output_layer = nn.Linear(self.embedding_dim, action_size)
    
    def forward(self, x):
        # Projection
        x = self.projection(x)

        # Attention
        if HAS_ATTENTION:
            x, _ = self.attention(x,x,x)
            x = self.relu(x)
        
        # Hidden layer 1 - no residual needed
        h1 = self.relu(self.hidden_layers[0](x))
        h1 = self.dropout(h1)

        # Rest of the hidden layers
        h = h1
        for i in range(1, self.num_hidden_layers):
            h = self.relu(self.hidden_layers[i](h + x))
            h = self.dropout(h)
        
        # Output layer
        out = self.output_layer(h)
        return out