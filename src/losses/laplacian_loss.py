import torch

def laplacian_loss(components, n_fft=256, hop_length=128, **kwargs):
    """
    Computes a loss based on the Laplacian (second derivative) of the frequency spectrum
    of the components. Penalizes roughness in the frequency domain.

    Args:
        components (torch.Tensor): [batch_size, n_components, signal_length]
        n_fft (int): FFT window size for STFT/FFT.
        hop_length (int): Hop length for STFT (not used for FFT).

    Returns:
        torch.Tensor: Scalar loss value (mean Laplacian magnitude across all components and batch).
    """
    # Reshape to [batch_size * n_components, signal_length]
    x = components.reshape(-1, components.shape[-1])

    # Compute FFT magnitude spectrum: [B*C, freq_bins]
    spectrum = torch.fft.rfft(x, n=n_fft)
    mag = spectrum.abs()

    # Compute discrete Laplacian (second difference) along frequency axis
    laplacian = mag[:, 2:] - 2 * mag[:, 1:-1] + mag[:, :-2]  # [B*C, freq_bins-2]

    # Penalize the mean squared Laplacian (smoothness)
    loss = (laplacian ** 2).mean()
    return loss