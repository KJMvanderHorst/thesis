import torch
import torch.nn.functional as F

def ssft_coherence_loss(components, n_fft=256, hop_length=128, win_length=None, **kwargs):
    """
    Differentiable STFT-based approximation of wavelet coherence.
    
    Args:
        components: Tensor of shape (batch_size, n_components, signal_length)
        n_fft, hop_length, win_length: STFT parameters for time-frequency representation

    Returns:
        Scalar Tensor: average pairwise coherence-like similarity (to penalize).
    """
    batch_size, n_components, signal_length = components.shape
    device = components.device
    win_length = win_length or n_fft
    window = torch.hann_window(win_length).to(device)

    # Reshape for batch-wise STFT
    signals = components.view(batch_size * n_components, signal_length)

    # Compute STFT: (B * C, freq, time)
    stfts = torch.stft(signals, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                       window=window, return_complex=True)
    mags = torch.abs(stfts)  # shape: (B * C, freq, time)

    # Normalize to unit norm for cosine similarity
    mags_flat = mags.reshape(batch_size, n_components, -1)  # (B, C, F*T)
    mags_flat = F.normalize(mags_flat, p=2, dim=-1)       # unit L2 norm

    # Compute pairwise cosine similarity
    # Result shape: (B, C, C)
    sim_matrix = torch.matmul(mags_flat, mags_flat.transpose(1, 2))

    # Only use upper triangle (i < j)
    triu_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1)
    pairwise_sims = sim_matrix[triu_mask.bool()]
    
    # Return average similarity (lower is better — less coherence)
    return pairwise_sims.mean()

def fft_coherence_loss(components, **kwargs):
    """
    Computes a coherence loss using normal FFT instead of STFT.

    Args:
        components: Tensor of shape (batch_size, n_components, signal_length)

    Returns:
        Scalar Tensor: average pairwise coherence-like similarity (to penalize).
    """
    batch_size, n_components, signal_length = components.shape
    device = components.device

    # Compute FFT: (B, C, F)
    fft_results = torch.fft.rfft(components, dim=-1)  # Only positive frequencies
    mags = torch.abs(fft_results)  # Magnitude spectrum: (B, C, F)

    # Normalize to unit norm for cosine similarity
    mags_flat = mags.reshape(batch_size, n_components, -1)  # (B, C, F)
    mags_flat = F.normalize(mags_flat, p=2, dim=-1)  # Unit L2 norm

    # Compute pairwise cosine similarity
    # Result shape: (B, C, C)
    sim_matrix = torch.matmul(mags_flat, mags_flat.transpose(1, 2))

    # Only use upper triangle (i < j)
    triu_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1)
    pairwise_sims = sim_matrix[triu_mask.bool()]

    # Return average similarity (lower is better — less coherence)
    return pairwise_sims.mean()

def complex_stft_similarity_loss(components, n_fft=256, hop_length=128, win_length=None, **kwargs):
    """
    Computes a differentiable dissimilarity loss using complex STFT inner products.

    Args:
        components: Tensor of shape (B, C, T) - decomposed components

    Returns:
        Scalar Tensor: average pairwise similarity (lower = better)
    """
    B, C, T = components.shape
    device = components.device
    win_length = win_length or n_fft
    window = torch.hann_window(win_length).to(device)

    # Reshape for batch STFT: (B*C, T)
    signals = components.reshape(B * C, T)

    # Complex STFT: (B*C, F, T')
    stfts = torch.stft(signals, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, return_complex=True)

    # Flatten frequency-time into a single vector: (B, C, F*T')
    stfts_flat = stfts.reshape(B, C, -1)

    # Normalize complex vectors using Hermitian norm
    norms = torch.sqrt(torch.sum(torch.abs(stfts_flat) ** 2, dim=-1, keepdim=True) + 1e-8)
    stfts_normed = stfts_flat / norms  # (B, C, F*T')

    # Compute complex inner products (cosine similarity w/ phase)
    sim_matrix = torch.matmul(stfts_normed, stfts_normed.transpose(1, 2).conj())  # (B, C, C)
    real_sims = sim_matrix.real  # discard imaginary part

    # Only upper triangle (i < j)
    triu_mask = torch.triu(torch.ones_like(real_sims), diagonal=1)
    pairwise_sims = real_sims[triu_mask.bool()]

    return pairwise_sims.mean()
