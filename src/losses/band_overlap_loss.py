import torch

def band_overlap_loss(components, **kwargs):
    """
    Compute the band overlap loss to maximize the overlap of frequency content
    within the designated frequency bands.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        frequency_bands (torch.Tensor): Frequency bands of shape [batch_size, n_bands, 2], where each band is (fmin, fmax).

    Returns:
        torch.Tensor: The band overlap loss value averaged over the batch.
    """
    if 'frequency_bands' not in kwargs:
        raise ValueError("Missing required argument 'frequency_bands' in kwargs.")
    
    frequency_bands = kwargs['frequency_bands']
    batch_size, n_components, signal_length = components.shape
    n_bands = frequency_bands.shape[1]

    # Compute the frequency axis for the FFT
    freqs = torch.fft.fftfreq(signal_length, d=1.0).to(components.device)  # Shape: [signal_length]

    # Compute the power spectrum for all components
    power_spectrum = torch.abs(torch.fft.fft(components, dim=-1))**2  # Shape: [batch_size, n_components, signal_length]

    # Expand frequency axis for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, signal_length]

    # Extract fmin and fmax for all bands
    fmin = frequency_bands[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, n_bands, 1]
    fmax = frequency_bands[:, :, 1].unsqueeze(-1)  # Shape: [batch_size, n_bands, 1]

    # Compute masks for frequencies within the bands
    within_band_mask = ((freqs >= fmin) & (freqs <= fmax)).float()  # Shape: [batch_size, n_bands, signal_length]

    # Expand power spectrum for broadcasting
    power_spectrum = power_spectrum.unsqueeze(2)  # Shape: [batch_size, n_components, 1, signal_length]

    # Compute weighted overlap for all bands and components
    weighted_overlap = (within_band_mask.unsqueeze(1) * power_spectrum).sum(dim=-1)  # Shape: [batch_size, n_components, n_bands]

    # Sum overlap across all bands
    band_overlap = weighted_overlap.sum(dim=-1)  # Shape: [batch_size, n_components]

    # Average the band overlap over all components and the batch
    band_overlap_loss_value = -band_overlap.mean()  # Negative to minimize the overlap

    # Return the scaled loss as a PyTorch tensor
    return band_overlap_loss_value