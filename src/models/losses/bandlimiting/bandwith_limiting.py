import numpy as np
from sklearn.cluster import KMeans

def compute_frequency_bands(signal, sampling_rate, n_bands=3):
    """
    Computes frequency bands by clustering the Fourier Transform of a signal
    based on both frequency and magnitude.

    Args:
        signal (np.ndarray): The input signal (1D array).
        sampling_rate (float): The sampling rate of the signal.
        n_bands (int): The number of frequency bands to compute.

    Returns:
        list of tuple: A list of frequency band ranges [(f_start, f_end), ...].
    """
    # Compute the Fourier Transform of the signal
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Use only the positive frequencies
    positive_freqs = freqs[freqs >= 0]
    positive_fft = fft[freqs >= 0]

    # Prepare 2D data for clustering: [frequency, magnitude]
    magnitudes = np.abs(positive_fft)
    clustering_data = np.column_stack((positive_freqs, magnitudes))

    # Cluster the frequencies using KMeans
    kmeans = KMeans(n_clusters=n_bands, random_state=42)
    clusters = kmeans.fit_predict(clustering_data)

    # Group frequencies into bands based on clusters
    frequency_bands = []
    for band in range(n_bands):
        band_freqs = positive_freqs[clusters == band]
        if len(band_freqs) > 0:
            f_start, f_end = band_freqs.min(), band_freqs.max()
            frequency_bands.append((f_start, f_end))

    return frequency_bands

def test_compute_frequency_bands_with_generated_signal(sampling_rate=1000, n_bands=3):
    """
    Tests the compute_frequency_bands function using a generated synthetic signal.

    Args:
        sampling_rate (float): Sampling rate of the synthetic signal.
        n_bands (int): Number of frequency bands to compute.
    """
    # Generate a synthetic signal with multiple frequency components
    t = np.linspace(0, 1, sampling_rate, endpoint=False)  # 1 second of data
    signal = (
        np.sin(2 * np.pi * 1 * t) +  # 50 Hz component
        np.sin(2 * np.pi * 2 * t) +  # 120 Hz component
        np.sin(2 * np.pi * 3 * t) +  # 300 Hz component
        np.sin(2 * np.pi * 151 * t) +  # 50 Hz component
        np.sin(2 * np.pi * 152 * t) +  # 120 Hz component
        np.sin(2 * np.pi * 153 * t) + # 300 Hz component
        np.sin(2 * np.pi * 300 * t) +  # 50 Hz component
        np.sin(2 * np.pi * 299 * t) +  # 120 Hz component
        np.sin(2 * np.pi * 298 * t)  # 300 Hz component
    )

    # Compute frequency bands
    frequency_bands = compute_frequency_bands(signal, sampling_rate, n_bands)

    # Print the results
    print("Computed Frequency Bands:", frequency_bands)

# Example usage
if __name__ == "__main__":

    SAMPLING_RATE = 1000  # Replace with the actual sampling rate of your dataset
    test_compute_frequency_bands_with_generated_signal(sampling_rate=1000, n_bands=3)