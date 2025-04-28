from src.models.losses import wavelet_coherence_loss, mse_loss, band_leakage_loss

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
        "wavelet_coherence": wavelet_coherence_loss,
        "mse": mse_loss,
        "band_leakage": band_leakage_loss,
    }

    total_loss = 0.0

    for loss_name in loss_list:
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}")
        # Call the appropriate loss function with **kwargs
        loss_fn = loss_functions[loss_name]
        #switch case for loss function
        total_loss += loss_fn(predicted, **kwargs)
    return total_loss
