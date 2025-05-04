from torch import nn
from src.models.RRCNN_layer import RRCNNLayer
from src.models.multi_scale_rrcnn_layer import MultiScaleRRCNNLayer
import torch

class RRCNNDecomposer(nn.Module):
    def __init__(self, n_components=2):
        super().__init__()
        self.imf_extractors = nn.ModuleList([
            RRCNNLayer() for _ in range(n_components)
        ])

    def forward(self, x):
        # x: [batch_size, 1, signal_length]
        components = torch.stack([layer(x) for layer in self.imf_extractors], dim=1)  # [batch_size, n_components, signal_length]
        return components
    


class MultiScaleRRCNNDecomposer(nn.Module):
    def __init__(self, n_components=2, conv_channels=16, kernel_sizes=(3, 5, 7)):
        """
        Multi-scale RRCNN Decomposer for signal decomposition.

        Args:
            n_components (int): Number of components to decompose the signal into.
            conv_channels (int): Number of output channels for each convolutional layer in the multi-scale RRCNN layer.
            kernel_sizes (tuple): A tuple of kernel sizes for multi-scale convolution.
        """
        super().__init__()
        self.imf_extractors = nn.ModuleList([
            MultiScaleRRCNNLayer(conv_channels=conv_channels, kernel_sizes=kernel_sizes)
            for _ in range(n_components)
        ])

    def forward(self, x):
        """
        Forward pass for the multi-scale RRCNN decomposer.

        Args:
            x (torch.Tensor): Input signal of shape [batch_size, 1, signal_length].

        Returns:
            torch.Tensor: Decomposed components of shape [batch_size, n_components, signal_length].
        """
        # Apply each multi-scale RRCNN layer to the input signal
        components = torch.stack([layer(x) for layer in self.imf_extractors], dim=1)  # [batch_size, n_components, signal_length]
        return components