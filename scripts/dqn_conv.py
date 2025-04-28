import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DQNConv(nn.Module):
    """
    A DQN that first applies a 2D convolution on the board (treated as a 2D image)
    and then passes the result through fully connected layers.

    board_size   : integer (n for an n x n board)
    action_size  : number of possible actions
    layer_sizes  : list of sizes for subsequent MLP layers
    kernel_size  : size of the convolution kernel (default 2, as requested)
    out_channels : number of output channels for the first conv layer
    """
    def __init__(self, board_size, action_size, layer_sizes, 
                 kernel_size=2, out_channels=8):
        super(DQNConv, self).__init__()
        self.board_size = int(math.sqrt(board_size))
        self.action_size = action_size

        # 1 input channel (board has values 0/1), out_channels can be tuned
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            padding=0  # No padding by default
        )

        # If board_size = 5 and kernel_size = 2, 
        # the output of the convolution in each dimension is (5 - 2 + 1) = 4
        # So the total flattened size is out_channels * 4 * 4 = 16 * 16 = 256 if out_channels=16.
        # More generally:
        conv_out_dim = (int(math.sqrt(board_size)) - kernel_size + 1)
        self.conv_output_size = out_channels * (conv_out_dim ** 2)

        # Build a sequence of Linear + ReLU layers
        layers = []
        input_dim = self.conv_output_size

        for hidden_size in layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        # Final linear layer that outputs Q-values for each action
        layers.append(nn.Linear(input_dim, action_size))
        
        # Wrap it all in a Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, board_size * board_size]
        # Reshape to [batch_size, 1 (channel), board_size, board_size]
        x = x.view(-1, 1, self.board_size, self.board_size)
        
        # Convolution layer
        x = self.conv(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.network(x)
        return x