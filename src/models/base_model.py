from torch import nn
from .RRCNN_layer import RRCNNLayer
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