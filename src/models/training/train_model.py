import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from src.models.base_model import RRCNNDecomposer
from src.data.dataset import SignalDataset
from src.models.losses.combined_loss import compute_combined_loss
from src.models.losses.bandlimiting.bandwidth_limiting import compute_frequency_bands

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50
EPOCHS = 10
loss_list = ['mse', 'wavelet_coherence', 'band_leakage']  # List of loss functions to use
LEARNING_RATE = 1e-3
N_COMPONENTS = 2  # Number of components for RRCNNDecomposer
DATA_PATH = "src/data/data_storage/composite_signals_20250422T181928.npz"  # Update with your dataset path
MODEL_SAVE_PATH = "models/rrcnn_decomposer_mock_test.pth"  # Specify a file name

def train():
    # Load the full dataset
    full_dataset = SignalDataset(DATA_PATH)

    # Split into train and test sets
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42
    )

    # Create train and test datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = RRCNNDecomposer(n_components=N_COMPONENTS).to(DEVICE)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    
    """# Check if 'band_leakage' loss is included
    if 'band_leakage' in loss_list:
        # Calculate bandwidths for all signals in the dataset using frequency spikes
        bandwidths = []
        for signal in full_dataset:
            if isinstance(signal, (tuple, list)):
                signal = signal[0]  # Extract the first element if it's a tuple or list
            # Ensure the signal is a 1D array
            signal = signal.squeeze()  # Removes dimensions of size 1 (e.g., (1, 1000) -> (1000,))
            # Use the frequency spikes function to calculate bandwidth
            bandwidth = compute_frequency_bands(signal)
            bandwidths.append(bandwidth)
    """

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        # Iterate over batches
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # If batch is a tuple or list, extract the signals
            if isinstance(batch, (tuple, list)):
                signals = batch[0]  # Extract the first element (signals)
            else:
                signals = batch  # If it's already a tensor, use it directly

            # Move signals to the device
            signals = signals.to(DEVICE)

            # Forward pass
            components = model(signals)

            # Compute loss
            loss = compute_combined_loss(components, signals, loss_list=loss_list)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log epoch loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()