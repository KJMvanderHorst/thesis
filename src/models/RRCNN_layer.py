import torch
import torch.nn as nn

class RRCNNLayer(nn.Module):
    def __init__(self, conv_channels=16, kernel_size=15):
        super().__init__()
        self.conv = nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv1d(conv_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # x: [batch_size, 1, signal_length]
        x = self.conv(x)      # -> [batch_size, conv_channels, signal_length]
        x = self.relu(x)
        x = self.out_conv(x)  # -> [batch_size, 1, signal_length]
        return x.squeeze(1)   # -> [batch_size, signal_length]