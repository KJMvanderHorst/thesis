import torch.fft

def band_leakage_loss(components, bands, sampling_rate):
    """
    Compute the bandwidth leakage loss by penalizing components that have frequencies outside their assigned bands.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        bands (list of tuples): List of frequency bands [(low1, high1), (low2, high2), ...] for each component.
        sampling_rate (float): Sampling rate of the signal.
    Returns:
        torch.Tensor: The bandwidth leakage loss value averaged over the batch.
    """
    # Ensure the number of bands matches the number of components
    if len(bands) != components.shape[1]:
        raise ValueError(f"Number of bands {len(bands)} must match the number of components {components.shape[1]}.")

    batch_size, n_components, signal_length = components.shape
    leakage_loss = 0.0

    # Compute the frequency resolution
    freq_resolution = sampling_rate / signal_length
    freqs = torch.fft.fftfreq(signal_length, d=1/sampling_rate).to(components.device)

    for i, (low, high) in enumerate(bands):
        # Compute the FFT of the i-th component
        component_fft = torch.fft.fft(components[:, i, :], dim=-1)

        # Create a mask for frequencies outside the allowed band
        band_mask = (freqs < low) | (freqs > high)

        # Compute the energy outside the band
        leakage_energy = torch.sum(torch.abs(component_fft[:, band_mask])**2, dim=-1)

        # Accumulate the leakage loss
        leakage_loss += leakage_energy.mean()

    # Return the average leakage loss over the batch
    return leakage_loss / n_components