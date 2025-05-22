import numpy as np
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d

SAMPLING_RATE = 1000 #TODO : move to config will fix later


def compute_bandwidths(signals, target_bands=2, sigma=2, num_thresholds=100):

    """
    Compute the bandwidths of a large array of signals.

    Args:
        signals (numpy.ndarray): Array of signals (shape: [n_signals, signal_length]).
        target_bands (int): Number of frequency bands to detect.
        sigma (float): Standard deviation for Gaussian smoothing.
        num_thresholds (int): Number of thresholds to try for band detection.

    Returns:
        list: A list of bandwidths for each signal. Each bandwidth is a list of tuples
              [(fmin1, fmax1), (fmin2, fmax2), ...] representing the detected bands.
    """


    # Check if the bandwidths are within the min and max frequencies of the signal
    min_freq = 10
    max_freq = 50


    bandwidths = []


    for signal in signals:
        # Step 1: Compute FFT
        n = len(signal)
        freqs = fftfreq(n, 1 / SAMPLING_RATE)[:n // 2]
        fft_magnitude = np.abs(fft(signal))[:n // 2]

        # Step 2: Apply Gaussian smoothing
        smoothed_magnitude = gaussian_filter1d(fft_magnitude, sigma=sigma)

        # Step 3: Detect frequency bands
        bands = find_frequency_bands(smoothed_magnitude, freqs, target_bands, num_thresholds)

        
        # Step 4: Check if the first band exceeds the max frequency
        if bands[0][1] > max_freq:
            full_band = (freqs[0], freqs[-1])
            bands = [full_band] * target_bands


        bandwidths.append(bands)
    return bandwidths

def find_frequency_bands(magnitude, freqs, target_bands, num_thresholds):
    """
    Detect frequency bands in the FFT magnitude spectrum.

    Args:
        magnitude (numpy.ndarray): FFT magnitude spectrum.
        freqs (numpy.ndarray): Corresponding frequency values.
        target_bands (int): Number of frequency bands to detect.
        num_thresholds (int): Number of thresholds to try.

    Returns:
        list: A list of tuples [(fmin, fmax), ...] representing the detected bands.
        It returns the bands in descending order of frequency.
    """

    # Ignore the DC component (0 Hz)
    magnitude = magnitude[1:]  # Exclude the first bin (DC component)
    freqs = freqs[1:]          # Exclude the corresponding frequency (0 Hz)
    thresholds = np.linspace(np.min(magnitude), np.max(magnitude), num_thresholds) # ascending order, important!!!

    for thresh in thresholds:
        mask = magnitude > thresh
        regions = []
        in_band = False
        start = None
        for i, active in enumerate(mask):
            if active and not in_band:
                start = i
                in_band = True
            elif not active and in_band:
                regions.append((freqs[start], freqs[i - 1]))
                in_band = False
        if in_band:
            regions.append((freqs[start], freqs[-1]))
        if len(regions) == target_bands:
            #sort the bands in descending order so that the first band is the lowest frequency
            regions.sort(key=lambda x: x[1], reverse=False)
            return regions
    full_band = (freqs[0], freqs[-1])
    return [full_band] * target_bands

