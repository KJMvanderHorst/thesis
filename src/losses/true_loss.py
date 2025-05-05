import torch.nn.functional as F

def true_loss(components, **kwargs):
    """
    Compute the Mean Squared Error (MSE) loss between the predicted components
    and the actual ground truth components.

    Args:
        components (torch.Tensor): Predicted components of shape [batch_size, n_components, signal_length].
        kwargs: Additional arguments, including 'ground_truth' which is the actual components.
        ground_truth (torch.Tensor): Ground truth components of shape [batch_size, n_components, signal_length].

    Returns:
        torch.Tensor: The MSE loss value averaged over the batch.
    """
    if 'ground_truth' not in kwargs:
        raise ValueError("Missing required argument 'ground_truth' in kwargs.")
    ground_truth = kwargs['ground_truth']

    # Ensure the shapes of components and ground_truth match
    if components.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: components shape {components.shape} and ground_truth shape {ground_truth.shape} must match.")

    # Compute MSE loss between the predicted components and the ground truth components
    loss = F.mse_loss(components, ground_truth, reduction='mean')  # Mean reduction over batch, components, and signal length
    return loss