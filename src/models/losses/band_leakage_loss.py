import torch
from src.models.losses.loss_config import BAND_LEAKAGE_PARAM

def band_leakage_loss(components, frequency_bands):
    """
    Compute the band leakage loss based on the frequency content outside the designated frequency bands.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        frequency_bands (torch.Tensor): Frequency bands of shape [batch_size, n_bands, 2], where each band is (fmin, fmax).

    Returns:
        torch.Tensor: The band leakage loss value averaged over the batch.
    """

    batch_size, n_components, signal_length = components.shape
    n_bands = frequency_bands.shape[1]

    # Compute the frequency axis for the FFT
    freqs = torch.fft.fftfreq(signal_length, d=1.0).to(components.device)  # Frequency values (normalized)

    # Initialize band leakage loss
    band_leakage = torch.zeros(batch_size, n_components).to(components.device)

    # Compute band leakage for each component
    for i in range(n_components):
        # Compute the power spectrum of the i-th component
        power_spectrum = torch.abs(torch.fft.fft(components[:, i, :], dim=-1))**2  # Shape: [batch_size, signal_length]

        # Compute the band leakage for each frequency band
        for j in range(n_bands):
            fmin, fmax = frequency_bands[:, j, 0], frequency_bands[:, j, 1]  # Extract band boundaries

            # Create masks for frequencies outside the band
            below_band_mask = (freqs < fmin.unsqueeze(1)).float()  # Frequencies below fmin
            above_band_mask = (freqs > fmax.unsqueeze(1)).float()  # Frequencies above fmax

            # Compute the distance from the band for out-of-band frequencies
            below_distance = (fmin.unsqueeze(1) - freqs) * below_band_mask  # Distance below fmin
            above_distance = (freqs - fmax.unsqueeze(1)) * above_band_mask  # Distance above fmax

            # Combine the masks and distances
            out_of_band_distance = below_distance + above_distance  # Shape: [batch_size, signal_length]

            # Compute the weighted leakage (distance * power spectrum)
            weighted_leakage = (out_of_band_distance * power_spectrum).sum(dim=-1)  # Sum over frequencies

            # Accumulate the leakage for this band
            band_leakage[:, i] += weighted_leakage

    # Average the band leakage over all components and the batch
    band_leakage_loss_value = band_leakage.mean()

    return band_leakage_loss_value * BAND_LEAKAGE_PARAM
