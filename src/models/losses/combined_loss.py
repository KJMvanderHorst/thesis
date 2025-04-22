from .mse_loss import mse_loss
from .wavelet_coherence_loss import compute_wavelet_coherence

LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'wavelet_coherence': compute_wavelet_coherence
}

def compute_combined_loss(output, target, loss_list):
    total_loss = 0.0
    for name in loss_list:
        loss_fn = LOSS_FUNCTIONS.get(name)
        if loss_fn is None:
            raise ValueError(f"Loss '{name}' is not defined.")
        total_loss += loss_fn(output, target)
    return total_loss
