import torch
import torch.nn.functional as F

def frequency_reconstruction_loss(components, **kwargs):
    """
    Compute a reconstruction loss in the frequency domain using the power spectrum.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        kwargs: Additional arguments, including 'input_signal' which is the original signal.
        input_signal (torch.Tensor): The original input signal of shape [batch_size, 1, signal_length].
        This is used to compute the loss.

    Returns:
        torch.Tensor: The frequency-domain reconstruction loss value averaged over the batch.
    """
    if 'input_signal' not in kwargs:
        raise ValueError("Missing required argument 'input_signal' in kwargs.")
    input_signal = kwargs['input_signal']

    # Sum the components along the n_components dimension to reconstruct the signal
    reconstruction = components.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1, signal_length]

    # Ensure the reconstruction and input_signal have the same shape
    if reconstruction.shape != input_signal.shape:
        raise ValueError(f"Shape mismatch: reconstruction shape {reconstruction.shape} and input_signal shape {input_signal.shape} must match.")

    # Compute FFT of the input signal and the reconstruction
    input_fft = torch.fft.rfft(input_signal, dim=-1)  # Shape: [batch_size, 1, freq_bins]
    reconstruction_fft = torch.fft.rfft(reconstruction, dim=-1)  # Shape: [batch_size, 1, freq_bins]

    # Compute the power spectra (square of the magnitude spectra)
    input_power = torch.abs(input_fft) ** 2  # Shape: [batch_size, 1, freq_bins]
    reconstruction_power = torch.abs(reconstruction_fft) ** 2  # Shape: [batch_size, 1, freq_bins]

    # Normalize the power spectra to unit L2 norm
    input_power = F.normalize(input_power, p=2, dim=-1)  # Normalize along the frequency dimension
    reconstruction_power = F.normalize(reconstruction_power, p=2, dim=-1)

    # Compute the squared difference between the power spectra
    penalty = (reconstruction_power - input_power) ** 2  # Penalize squared difference

    # Average the penalty over the batch and frequency bins
    loss = penalty.mean(dim=-1)  # Shape: [batch_size, 1]
    loss = loss.mean()  # Final average over the batch

    return loss