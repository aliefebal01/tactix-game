import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    A DQN that takes a list of hidden layer sizes as parameter.
    Example layer_sizes = [50, 125] -> 2 hidden layers of sizes 50 and 125.
    """
    def __init__(self, state_size, action_size, layer_sizes):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Create a list of linear layers
        # E.g., if layer_sizes=[50,125], 
        # we want Linear(state_size, 50), then Linear(50,125), then Linear(125,action_size)
        
        # Build a sequence of Linear + ReLU (except for the output layer)
        layers = []
        input_dim = state_size
        
        for hidden_size in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        
        # Final layer: from the last hidden to the action output
        layers.append(nn.Linear(input_dim, action_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Outputs Q-values for each action