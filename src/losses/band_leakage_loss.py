import torch
import matplotlib.pyplot as plt

def band_leakage_loss(components, **kwargs):
    """
    Compute the band leakage loss based on the frequency content outside the designated frequency bands
    and plot the per-frequency contributions for all components in the batch.

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

    # Compute the frequency axis for the FFT (only positive frequencies for real-valued signals)
    freqs = torch.fft.rfftfreq(signal_length, d=1000).to(components.device)  # Shape: [signal_length // 2 + 1]

    # Compute the power spectrum for all components (only positive frequencies)
    power_spectrum = torch.abs(torch.fft.rfft(components, dim=-1))  # Shape: [batch_size, n_components, signal_length // 2 + 1]

    # Plot the components and their power spectrum
    for i in range(components.shape[0]):  # Limit to 5 examples for visualization
        plt.figure(figsize=(12, 6))

        # Plot the time-domain components
        plt.subplot(2, 1, 1)
        for j in range(components.shape[1]):
            plt.plot(components[i, j].cpu().detach().numpy(), label=f'Component {j+1}')
        plt.title(f'Components (Batch {i+1})')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot the power spectrum
        plt.subplot(2, 1, 2)
        for j in range(power_spectrum.shape[1]):
            plt.plot(freqs.squeeze().cpu().detach().numpy(), power_spectrum[i, j].cpu().detach().numpy(), label=f'Component {j}+1')
    # Expand frequency axis for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, signal_length // 2 + 1]

    fmin = (frequency_bands[:, :, 0]).unsqueeze(-1)  # Shape: [batch_size, n_bands, 1]
    fmax = (frequency_bands[:, :, 1]).unsqueeze(-1)

    # Compute masks for frequencies outside the bands
    below_band_mask = (freqs < fmin).float()  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    above_band_mask = (freqs > fmax).float()  # Shape: [batch_size, n_bands, signal_length // 2 + 1]

    

    # Compute distances for out-of-band frequencies
    below_distance = (fmin - freqs) * below_band_mask  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    above_distance = (freqs - fmax) * above_band_mask  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    out_of_band_distance = below_distance + above_distance  # Shape: [batch_size, n_bands, signal_length // 2 + 1]

    # Expand power spectrum for broadcasting
    power_spectrum = power_spectrum.unsqueeze(2)  # Shape: [batch_size, n_components, 1, signal_length // 2 + 1]

    # Normalize the power spectrum to prevent it from dominating
    normalized_power_spectrum = power_spectrum / power_spectrum.max()  # Normalize to [0, 1]

    # Compute weighted leakage for all bands and components
    weighted_leakage = (out_of_band_distance.unsqueeze(1)**2 * power_spectrum).sum(dim=-1)  # Shape: [batch_size, n_components, n_bands]

    # Sum leakage across all bands
    band_leakage = weighted_leakage.sum(dim=-1)  # Shape: [batch_size, n_components]

    # Average the band leakage over all components and the batch
    band_leakage_loss_value = band_leakage.mean()

    # Return the scaled loss as a PyTorch tensor
    return band_leakage_loss_value