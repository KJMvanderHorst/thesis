import torch

def wavelet_coherence_loss(components, **kwargs):
    """
    Computes the average pairwise wavelet coherence between decomposed components.

    Args:
        components: Tensor of shape (batch_size, n_components, signal_length)
            representing the decomposed components.

    Returns:
        Scalar Tensor: average wavelet coherence penalty.
    """
    batch_size, n_components, signal_length = components.shape
    total_coherence = 0.0
    pair_count = 0

    for b in range(batch_size):
        for i in range(n_components):
            for j in range(i + 1, n_components):
                sig1 = components[b, i]
                sig2 = components[b, j]

                # Compute the cross-spectrum and power spectral densities
                f_sig1 = torch.fft.fft(sig1)
                f_sig2 = torch.fft.fft(sig2)
                cross_spectrum = torch.abs(f_sig1 * torch.conj(f_sig2)) ** 2
                psd_sig1 = torch.abs(f_sig1) ** 2
                psd_sig2 = torch.abs(f_sig2) ** 2

                # Compute coherence
                coherence = cross_spectrum / (psd_sig1 * psd_sig2 + 1e-8)  # Add small epsilon for numerical stability
                total_coherence += coherence.mean()
                pair_count += 1

    average_coherence = total_coherence / max(pair_count, 1)
    return average_coherence