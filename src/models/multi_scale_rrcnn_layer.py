import torch
import torch.nn as nn

class MultiScaleRRCNNLayer(nn.Module):
    def __init__(self, conv_channels=16, kernel_sizes=(5, 15, 30)):
        """
        Multi-scale RRCNN layer with multiple convolutional layers of different kernel sizes.

        Args:
            conv_channels (int): Number of output channels for each convolutional layer.
            kernel_sizes (tuple): A tuple of kernel sizes for multi-scale convolution.
        """
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=ks,
                padding='same',  # Use 'same' padding to keep the output size same as input
            )
            for ks in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv1d(conv_channels * len(kernel_sizes), 1, kernel_size=1)

    def forward(self, x):
        # x: [batch_size, 1, signal_length]
        multi_scale_features = []
        for conv in self.convs:
            feature = self.relu(conv(x))  # Apply each convolution followed by ReLU
            multi_scale_features.append(feature)
        
        # Concatenate features along the channel dimension
        x = torch.cat(multi_scale_features, dim=1)  # -> [batch_size, conv_channels * len(kernel_sizes), signal_length]
        x = self.out_conv(x)  # -> [batch_size, 1, signal_length]
        return x.squeeze(1)   # -> [batch_size, signal_length]