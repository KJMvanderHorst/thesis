import torch
import numpy as np
import pywt
from scipy.signal import coherence
from ...data.data_generation.data_config import SAMPLING_RATE

def compute_wavelet_coherence(components, input_signal):
    """
    Computes the average pairwise wavelet coherence between decomposed components.

    Args:
        components: Tensor of shape (batch_size, n_components, signal_length)
            representing the decomposed components.

    Returns:
        Scalar Tensor: average wavelet coherence penalty.
    """
    batch_size, n_components, signal_length = components.shape
    components_np = components.detach().cpu().numpy()
    total_coherence = 0.0
    pair_count = 0

    for b in range(batch_size):
        for i in range(n_components):
            for j in range(i + 1, n_components):
                sig1 = components_np[b, i]
                sig2 = components_np[b, j]

                # Use scipy's coherence as a proxy
                f, Cxy = coherence(sig1, sig2, fs= SAMPLING_RATE, nperseg=256)
                total_coherence += np.mean(Cxy)
                pair_count += 1

    average_coherence = total_coherence / max(pair_count, 1)
    return torch.tensor(average_coherence, device=components.device, dtype=components.dtype)