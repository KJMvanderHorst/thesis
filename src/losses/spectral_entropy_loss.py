import torch

def spectral_entropy_loss(components, n_fft=256, hop_length=128, eps=1e-8, **kwargs):
    """
    Vectorized spectral entropy loss: penalizes high entropy in the STFT of each component,
    encouraging local narrowband behaviour.

    Args:
        components (torch.Tensor): [batch_size, n_components, signal_length]
        n_fft (int): FFT window size for STFT.
        hop_length (int): Hop length for STFT.
        eps (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: Scalar loss value (mean entropy across all components and batch).
    """
    # Reshape to [batch_size * n_components, signal_length]
    x = components.reshape(-1, components.shape[-1])

    # Compute STFT: output shape [B*C, freq_bins, time_frames]
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    power = stft.abs() ** 2  # [B*C, freq_bins, time_frames]

    # Normalize over frequency bins for each time frame
    power_sum = power.sum(dim=1, keepdim=True) + eps  # [B*C, 1, time_frames]
    prob = power / power_sum  # [B*C, freq_bins, time_frames]

    # Compute entropy for each time frame
    entropy = -torch.sum(prob * torch.log(prob + eps), dim=1)  # [B*C, time_frames]

    # Mean over time frames, then mean over all components and batch
    mean_entropy = entropy.mean()

    return mean_entropy