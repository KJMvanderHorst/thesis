import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os

training_run_losses = []

def band_leakage_loss(components, **kwargs):
    """
    Compute the band leakage loss based on the frequency content outside the designated frequency bands
    and plot the per-frequency contributions for all components in the batch.

    Args:
        components (torch.Tensor): Decomposed components of shape [batch_size, n_components, signal_length].
        frequency_bands (torch.Tensor): Frequency bands of shape [batch_size, n_bands, 2], where each band is (fmin, fmax).

    Returns:
        torch.Tensor: The band leakage loss value averaged over the batch.
    """
    if 'frequency_bands' not in kwargs:
        raise ValueError("Missing required argument 'frequency_bands' in kwargs.")
    
    frequency_bands = kwargs['frequency_bands']
    batch_size, n_components, signal_length = components.shape
    n_bands = frequency_bands.shape[1]

    # Mean-center the components along the time axis
    components = components - components.mean(dim=-1, keepdim=True)

    # Compute the frequency axis for the FFT (only positive frequencies for real-valued signals)
    freqs = torch.fft.rfftfreq(signal_length, d=1/1000).to(components.device)  # Shape: [signal_length // 2 + 1]

    # Compute the power spectrum for all components (only positive frequencies)
    power_spectrum = torch.abs(torch.fft.rfft(components, dim=-1))  # Shape: [batch_size, n_components, signal_length // 2 + 1]
    power_spectrum = power_spectrum ** 2  # Power spectrum: square of the magnitude
    power_spectrum = F.normalize(power_spectrum, p=2, dim=-1)  # Normalize to unit L2 norm

    # Expand frequency axis for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, signal_length // 2 + 1]

    margin = 2  # Example: Expand bands by 2 Hz on each side
    fmin = (frequency_bands[:, :, 0] - margin).unsqueeze(-1)  # Shape: [batch_size, n_bands, 1]
    fmax = (frequency_bands[:, :, 1] + margin).unsqueeze(-1)

    # Compute masks for frequencies outside the bands
    below_band_mask = (freqs < fmin).float()  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    above_band_mask = (freqs > fmax).float()  # Shape: [batch_size, n_bands, signal_length // 2 + 1]

    
    # Compute distances for out-of-band frequencies
    below_distance = (fmin - freqs) * below_band_mask  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    above_distance = (freqs - fmax) * above_band_mask  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    out_of_band_distance = below_distance + above_distance  # Shape: [batch_size, n_bands, signal_length // 2 + 1]
    
    # Compute the contribution of each frequency to the loss per component
    frequency_contributions = (out_of_band_distance * power_spectrum)  #hape: [batch_size, n_components, signal_length // 2 + 1]

    # Plot the out-of-band distance and power spectrum for each component
    """for i in range(frequency_contributions.shape[0]):  # Iterate over batch elements
        for j in range(frequency_contributions.shape[1]):  # Iterate over components
            plt.figure(figsize=(12, 12))

            # Plot the out-of-band distance for the current component
            plt.subplot(2, 1, 1)
            plt.plot(freqs.squeeze().cpu().detach().numpy(),
                     out_of_band_distance[0][j].cpu().detach().numpy(),
                     label=f'Out-of-Band Distance (Component {j+1})')
            plt.title(f'Out-of-Band Distance (Batch {i+1}, Component {j+1})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Out-of-Band Distance')
            plt.legend()

            # Plot the power spectrum for the current component
            plt.subplot(2, 1, 2)
            plt.plot(freqs.squeeze().cpu().detach().numpy(),
                     power_spectrum[i, j].cpu().detach().numpy(),
                     label=f'Power Spectrum (Component {j+1})')
            plt.title(f'Power Spectrum (Batch {i+1}, Component {j+1})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.legend()

            plt.tight_layout()
            plt.show()"""

    # Plot the frequency contributions to the loss per component
    """for i in range(frequency_contributions.shape[0]):  # Iterate over batch elements
        for j in range(frequency_contributions.shape[1]):  # Iterate over components
            print(f"Batch {i+1}, Component {j+1} Frequency Contributions:")
            plt.figure(figsize=(8, 4))

            # Plot the predicted component in the time domain
            plt.subplot(2, 1, 1)
            plt.plot(components[i, j].cpu().detach().numpy(), label=f'Predicted Component {j+1}')
            plt.title(f'Predicted Component (Batch {i+1}, Component {j+1})')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()

            # Plot the frequency contributions for the current component
            plt.plot(freqs.squeeze().cpu().detach().numpy(),
                    frequency_contributions[i, j].cpu().detach().numpy(),
                    label=f'Frequency Contribution (Component {j+1})')

            # Add shaded regions for the frequency bands
            for band in range(frequency_bands.shape[1]):
                fmin = frequency_bands[i, band, 0].cpu().detach().numpy()
                fmax = frequency_bands[i, band, 1].cpu().detach().numpy()
                plt.axvspan(fmin, fmax, color='gray', alpha=0.3, label=f'Band {band+1}' if j == 0 else None)

            plt.title(f'Frequency Contributions to Loss (Batch {i+1}, Component {j+1})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Contribution to Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()"""
    
    # Compute the average band leakage loss across the batch based on the frequency contributions
    band_leakage_loss_value = frequency_contributions.sum(dim=-1) # Shape: [batch_size, n_components]

    # Compute the average band leakage loss across the batch and store per-component losses
    band_leakage_per_component = band_leakage_loss_value.mean(dim=0)
    training_run_losses.append(band_leakage_per_component.cpu().detach().numpy())

    #print(f"Band Leakage Loss Value: {band_leakage_loss_value}")
    band_leakage_loss_value = band_leakage_loss_value.mean(dim=-1)  # Average over the components
    band_leakage_loss_value = band_leakage_loss_value.mean()  # Average over the batch


    # Return the scaled loss as a PyTorch tensor
    return band_leakage_loss_value

def save_training_run_losses(filename='training_run_losses.npz'):
    """
    Save all accumulated per-component losses for the training run to a .npz file.

    Args:
        filename (str): The name of the file to save the losses.
    """
    global training_run_losses

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define the target directory for saving the file
    data_dir = os.path.join(project_root, "src/losses/band_leakage_loss_data")

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Construct the full file path
    data_path = os.path.join(data_dir, filename)

    # Save the training run losses to the .npz file
    np.savez(data_path, training_run_losses=np.array(training_run_losses))
    print(f"Saved training run losses to {data_path}")

    # Clear the list after saving
    training_run_losses.clear()
    print("Cleared training run losses list.")

def clear_training_run_losses():
    """
    Clear the accumulated per-component losses for the training run.
    """
    global training_run_losses
    training_run_losses.clear()
    print("Cleared training run losses list.")