import torch
import torch.fft

def energy_ordering_loss(components, sampling_rate=1000, **kwargs):
    """
    Compute a loss that penalizes the model if the components are not ordered
    in ascending order of frequency content (bandwidth).

    Args:
        components (torch.Tensor): Tensor of shape [batch_size, n_components, signal_length].
                                   Each component is a signal.
        sampling_rate (int): Sampling rate of the signals (in Hz).

    Returns:
        torch.Tensor: Scalar loss value. High if components are not ordered by frequency content.
    """
    batch_size, n_components, signal_length = components.shape

    # Compute FFT of each component
    fft_result = torch.fft.rfft(components, dim=-1)  # Shape: [batch_size, n_components, freq_bins]
    fft_magnitude = torch.abs(fft_result)  # Magnitude spectrum

    # Compute frequency bins
    freq_bins = torch.fft.rfftfreq(signal_length, d=1.0 / sampling_rate).to(components.device)  # Shape: [freq_bins]

    # Compute the "center of mass" of the frequency content (weighted average frequency)
    # This represents the dominant frequency of each component
    dominant_frequencies = torch.sum(fft_magnitude * freq_bins, dim=-1) / torch.sum(fft_magnitude, dim=-1)
    # Shape: [batch_size, n_components]

    # Compute the differences between consecutive dominant frequencies
    frequency_differences = dominant_frequencies[:, 1:] - dominant_frequencies[:, :-1]  # Shape: [batch_size, n_components - 1]

    # Penalize negative differences (i.e., when frequencies are not in ascending order)
    penalties = torch.clamp(-frequency_differences, min=0)  # Negative differences become positive penalties

    # Compute the total loss as the mean penalty across the batch and components
    loss = penalties.mean()

    return loss