import torch
import torch.nn.functional as F

def frequency_gradient_loss(components, n_fft=256, **kwargs):
    """
    Penalizes the first derivative (gradient) of the frequency spectrum of the components,
    encouraging spectral sharpness (i.e., discouraging overly smooth or smeared spectra).

    Args:
        components (torch.Tensor): [batch_size, n_components, signal_length]
        n_fft (int): FFT window size.

    Returns:
        torch.Tensor: Scalar loss value (mean squared gradient across all components and batch).
    """
    # Reshape to [batch_size * n_components, signal_length]
    x = components.reshape(-1, components.shape[-1])

    # Compute STFT power spectrum: [B*C, freq_bins, time_frames]
    stft = torch.stft(x, n_fft=n_fft, return_complex=True)
    power = stft.abs() ** 2

    # Normalize across frequency axis for each time frame
    power = power / (power.sum(dim=1, keepdim=True) + 1e-8)

    # Compute finite difference (frequency gradient)
    grad_freq = power[:, 1:, :] - power[:, :-1, :]  # [B*C, freq_bins-1, time_frames]

    # Penalize large changes (mean squared gradient)
    loss = grad_freq.pow(2).mean()
    return loss