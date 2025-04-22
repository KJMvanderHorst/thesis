from losses.reconstruction_loss import mse_loss
from losses.wavelet_coherence_loss import wavelet_coherence_loss

LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'wavelet_coherence': wavelet_coherence_loss,
}

def compute_combined_loss(output, target, loss_list):
    total_loss = 0.0
    for name in loss_list:
        loss_fn = LOSS_FUNCTIONS.get(name)
        if loss_fn is None:
            raise ValueError(f"Loss '{name}' is not defined.")
        total_loss += loss_fn(output, target)
    return total_loss
