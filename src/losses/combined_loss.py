from src.losses.wavelet_coherence_loss import compute_wavelet_coherence
from src.losses.mse_loss import mse_loss
from src.losses.band_leakage_loss import band_leakage_loss

def compute_combined_loss(predicted, loss_list, **kwargs):
    """
    Compute a combined loss based on the specified loss functions.

    Args:
        predicted (torch.Tensor): Predicted outputs from the model.
        loss_list (list): List of loss function names to use.
        kwargs: Additional arguments for the loss functions.

    Returns:
        torch.Tensor: Combined loss value.
    """
    loss_functions = {
        "wavelet_coherence": compute_wavelet_coherence,
        "mse": mse_loss,
        "band_leakage": band_leakage_loss,
    }

    total_loss = 0.0

    for loss_name in loss_list:
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}")
        # Call the appropriate loss function with **kwargs
        loss_fn = loss_functions[loss_name]

        total_loss += loss_fn(predicted, **kwargs) * kwargs['loss_weights'][loss_name]
    return total_loss
