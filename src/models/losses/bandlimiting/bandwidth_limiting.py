import numpy as np
from scipy.signal import find_peaks

def compute_frequency_bands(signal, sampling_rate, n_bands=3):
    """
    Computes frequency bands by detecting peaks in the Fourier Transform of a signal.

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
    positive_fft = np.abs(fft[freqs >= 0])  # Use the magnitude of the FFT

    # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(positive_fft, height=np.max(positive_fft) * 0.1)  # Peaks above 10% of max magnitude

    # Sort peaks by magnitude (descending) and select the top `n_bands`
    if len(peaks) > n_bands:
        sorted_peaks = sorted(peaks, key=lambda p: positive_fft[p], reverse=True)[:n_bands]
    else:
        sorted_peaks = peaks

    # Define frequency bands around each peak
    frequency_bands = []
    for peak in sorted_peaks:
        # Find the start and end of the band by looking for where the magnitude drops significantly
        band_start = peak
        band_end = peak
        threshold = positive_fft[peak] * 0.5  # 50% of the peak magnitude

        # Expand the band to the left
        while band_start > 0 and positive_fft[band_start] > threshold:
            band_start -= 1

        # Expand the band to the right
        while band_end < len(positive_fft) - 1 and positive_fft[band_end] > threshold:
            band_end += 1

        # Convert indices to frequencies
        f_start = positive_freqs[band_start]
        f_end = positive_freqs[band_end]
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
        np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
        np.sin(2 * np.pi * 120 * t) +  # 120 Hz component
        np.sin(2 * np.pi * 300 * t)  # 300 Hz component
    )

    # Compute frequency bands
    frequency_bands = compute_frequency_bands(signal, sampling_rate, n_bands)

    # Print the results
    print("Computed Frequency Bands:", frequency_bands)

# Example usage
if __name__ == "__main__":
    test_compute_frequency_bands_with_generated_signal(sampling_rate=1000, n_bands=3)