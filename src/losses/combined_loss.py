from src.losses.wavelet_coherence_loss import wavelet_coherence_loss
from src.losses.reconstruction_loss import reconstruction_loss
from src.losses.band_leakage_loss import band_leakage_loss
from src.losses.true_loss import true_loss
from src.losses.band_overlap_loss import band_overlap_loss

from src.losses.ssft_coherence import ssft_coherence_loss
from src.losses.frequency_reconstruction import frequency_reconstruction_loss
from src.losses.energy_ordering_loss import energy_ordering_loss
from src.losses.spectral_entropy_loss import spectral_entropy_loss
from src.losses.laplacian_loss import laplacian_loss


def compute_combined_loss(predicted, loss_list, **kwargs):
    """
    Compute a combined loss based on the specified loss functions.

    Args:
        predicted (torch.Tensor): Predicted outputs from the model.
        loss_list (list): List of loss function names to use.
        kwargs: Additional arguments for the loss functions.

    Returns:

        torch.Tensor: Combined loss value.
        per_loss_vals (dict): Dictionary containing individual loss values.

    """
    loss_functions = {
        "wavelet_coherence": wavelet_coherence_loss,
        "reconstruction": reconstruction_loss,
        "band_leakage": band_leakage_loss,
        "band_overlap": band_overlap_loss,

        "ssft_coherence": ssft_coherence_loss,
        "energy_ordering": energy_ordering_loss,
        "frequency_reconstruction": frequency_reconstruction_loss,
        "spectral_entropy": spectral_entropy_loss,
        "laplacian": laplacian_loss,
        "true_loss": true_loss
    }

    # Dictionary to store loss values for each loss function
    per_loss_vals = {loss_name : 0.0 for loss_name in loss_list}

    total_loss = 0.0

    for loss_name in loss_list:
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}")
        # Call the appropriate loss function with **kwargs
        loss_fn = loss_functions[loss_name]
        loss_val = loss_fn(predicted, **kwargs) * kwargs['loss_weights'][loss_name]
        
        per_loss_vals[loss_name] = loss_val.item()
        #print(f"Loss {loss_name}: {loss_val.item()}")
        total_loss += loss_val

    return total_loss, per_loss_vals
