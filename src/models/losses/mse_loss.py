import torch
import torch.nn.functional as F

from src.models.losses.loss_config import MSE_LOSS_PARAM

def mse_loss(components, input_signal):
    """
    Compute the Mean Squared Error (MSE) loss between the summed components (reconstruction)
    and the original input signal.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        input_signal (torch.Tensor): Original input signal of shape [batch_size, 1, signal_length].

    Returns:
        torch.Tensor: The MSE loss value averaged over the batch.
    """
    # Sum the components along the n_components dimension to reconstruct the signal
    reconstruction = components.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1, signal_length]

    # Ensure the reconstruction and input_signal have the same shape
    if reconstruction.shape != input_signal.shape:
        raise ValueError(f"Shape mismatch: reconstruction shape {reconstruction.shape} and input_signal shape {input_signal.shape} must match.")

    # Compute MSE loss between the reconstruction and the original input signal
    loss = F.mse_loss(reconstruction, input_signal, reduction='mean')  # Mean reduction over batch and signal length
    return loss * MSE_LOSS_PARAM

