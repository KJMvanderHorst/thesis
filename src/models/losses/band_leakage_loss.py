import torch
from src.models.losses.loss_config import BAND_LEAKAGE_PARAM

def band_leakage_loss(components, **kwargs):
    """
    Compute the band leakage loss based on the frequency content outside the designated frequency bands.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        frequency_bands (torch.Tensor): Frequency bands of shape [batch_size, n_bands, 2], where each band is (fmin, fmax).

    Returns:
        torch.Tensor: The band leakage loss value averaged over the batch.
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

    # Compute masks for frequencies outside the bands
    below_band_mask = (freqs < fmin).float()  # Shape: [batch_size, n_bands, signal_length]
    above_band_mask = (freqs > fmax).float()  # Shape: [batch_size, n_bands, signal_length]

    # Compute distances for out-of-band frequencies
    below_distance = (fmin - freqs) * below_band_mask  # Shape: [batch_size, n_bands, signal_length]
    above_distance = (freqs - fmax) * above_band_mask  # Shape: [batch_size, n_bands, signal_length]
    out_of_band_distance = below_distance + above_distance  # Shape: [batch_size, n_bands, signal_length]

    # Expand power spectrum for broadcasting
    power_spectrum = power_spectrum.unsqueeze(2)  # Shape: [batch_size, n_components, 1, signal_length]

    # Compute weighted leakage for all bands and components
    weighted_leakage = (out_of_band_distance.unsqueeze(1) * power_spectrum).sum(dim=-1)  # Shape: [batch_size, n_components, n_bands]

    # Sum leakage across all bands
    band_leakage = weighted_leakage.sum(dim=-1)  # Shape: [batch_size, n_components]

    # Average the band leakage over all components and the batch
    band_leakage_loss_value = band_leakage.mean()

    # Ensure BAND_LEAKAGE_PARAM is a tensor
    band_leakage_param_tensor = torch.tensor(BAND_LEAKAGE_PARAM, device=components.device, dtype=band_leakage_loss_value.dtype)

    # Return the scaled loss as a PyTorch tensor
    return band_leakage_loss_value * band_leakage_param_tensor