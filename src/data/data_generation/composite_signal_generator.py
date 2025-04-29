import numpy as np
from signals import *
from data_config import INTERMITTENCE_LOWER_BOUND, INTERMITTENCE_UPPER_BOUND, MIN_AMPLITUDE, MAX_AMPLITUDE, MIN_MODULATION_FREQUENCY,MAX_MODULATION_FREQUENCY, MAX_FFREQUENCY_DEVATION



"""
composite_signal_generator .py

This module provides a function to generate a synthetic dataset of signals
with various modulation types. The signals are generated based on specified
parameters such as frequency range, duration, and sample rate. The generated
signals can be used for testing and training machine learning models in
signal processing applications.

Classes:
    SyntheticSignalGenerator: A class to generate synthetic signals with various modulation types.
        Attributes:
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
            duration (float): Duration of the signal in seconds.
            signal_types (list): List of signal types to generate. Options include:
                'linear_am', 'sinusoidal_am', 'linear_fm', 'sinusoidal_fm',
                'amfm' and 'sine'.

"""

class SyntheticSignalGenerator:
    """
    A class to generate synthetic a dataset of signals with various modulation types.
    Attributes:
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        intermittence (float): Probability of generating an intermittent signal.
        overlap_factor (float): Factor to determine the overlap between segments.
        overlap_std (float): Standard deviation for the overlap adjustment.
        duration (float): Duration of the signal in seconds.
        signal_types (list): List of signal types to generate. Options include:
            'linear_am', 'sinusoidal_am', 'linear_fm', 'sinusoidal_fm',
            'amfm' and 'sine'.
    """
    def __init__(self, fmin, fmax, duration, signal_types, intermittence, overlap_factor, overlap_std):
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.signal_types = signal_types
        self.intermittence = intermittence
        self.overlap_factor = overlap_factor
        self.overlap_std = overlap_std

    def generate_signal(self, f0, bandwidth, k):
        """
        Generate a synthetic signal of signals with various modulation types.
        The signals are generated based on the specified parameters such as
        frequency range, duration, and sample rate.

        Args:
            k (int): Number of frequency segments.
            f0 (float): Starting frequency of the signal (Hz).
            bandwidth (float): Bandwidth of the signal (Hz).

        Returns:
            composite_signal (numpy.ndarray): The generated composite signal.
            components (list): A list of individual signal components.
        """
        # Generate frequency segments
        B_segments = self.generate_bandwidth(k, f0, bandwidth)
        components = []
        signals = []

        for i in range(k):
            # Generate the duration of the signal based on some probability of intermittence
            if np.random.uniform(0, 1) > self.intermittence:
                duration = self.duration
                t_start = 0  # No intermittence, signal starts at the beginning
            else:
                duration = np.random.uniform(INTERMITTENCE_LOWER_BOUND, INTERMITTENCE_UPPER_BOUND) * self.duration
                t_start = np.random.uniform(0, self.duration - duration)  # Random start time for intermittent signal

            # Randomly select a signal type from the provided list
            signal_type = np.random.choice(self.signal_types)

            starting_freq = np.random.uniform(0, 2 * np.pi)
            match signal_type:
                case 'linear_am':
                    freq = np.random.uniform(B_segments[i, 0], B_segments[i, 1])  # Pick a random frequency within the segment
                    am1 = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)
                    am2 = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)
                    am_max = np.max([am1, am2])
                    am_min = np.min([am1, am2])
                    signal = LinearAMSignal(b=am_max, a=am_min, fam=freq, phi=starting_freq, duration=duration)

                case 'sinusoidal_am':
                    freq = np.random.uniform(B_segments[i, 0], B_segments[i, 1])  # Pick a random frequency within the segment
                    modulating_freq_start = np.random.uniform(0, 2 * np.pi)
                    modulating_frequency = np.random.uniform(MIN_MODULATION_FREQUENCY, MAX_MODULATION_FREQUENCY)
                    signal = SinusoidalAMSignal(fs=modulating_frequency, phi_s=modulating_freq_start, fam=freq, phi=starting_freq, duration=duration)

                case 'linear_fm':
                    signal = LinearFMSignal(f0=B_segments[i][0], B=B_segments[i + 1] - B_segments[i], T=duration, phi=0)

                case 'sinusoidal_fm':
                    freq = np.random.uniform(B_segments[i, 0], B_segments[i, 1])  # Pick a random frequency within the segment
                    frequency_deviation = np.random.uniform(0, MAX_FFREQUENCY_DEVATION * freq)
                    modulating_frequency = np.random.uniform(MIN_MODULATION_FREQUENCY, MAX_MODULATION_FREQUENCY)
                    signal = SinusoidalFMSignal(fc=freq, fd=frequency_deviation, fm=modulating_frequency, phi=starting_freq, duration=duration)

                case 'amfm':
                    freq = np.random.uniform(B_segments[i, 0], B_segments[i, 1])  # Pick a random frequency within the segment
                    am1 = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)
                    am2 = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)
                    am_max = np.max([am1, am2])
                    am_min = np.min([am1, am2])
                    am = LinearAMSignal(b=am_max, a=am_min, fam=freq, phi=starting_freq, duration=duration)
                    fm = LinearFMSignal(f0=B_segments[i][0], B=B_segments[i + 1] - B_segments[i], T=duration, phi=0)
                    signal = AMFMSignal(am_signal=am, fm_signal=fm)

                case _:
                    freq = np.random.uniform(B_segments[i, 0], B_segments[i, 1])  # Pick a random frequency within the segment
                    signal = SineSignal(frequency=freq, phase=0, duration=duration)

            # Generate the signal and pad it to fit the total duration
            generated_signal = signal.generate()
            padded_signal = np.zeros(int(self.duration * SAMPLING_RATE))  # Create a zero array for the total duration
            t_start_idx = int(t_start * SAMPLING_RATE)
            t_end_idx = t_start_idx + len(generated_signal)
            padded_signal[t_start_idx:t_end_idx] = generated_signal  # Insert the intermittent signal

            signals.append(signal)

            components.append(padded_signal)

        # Sum all components to create the composite signal
        composite_signal = np.sum(components, axis=0)
        return composite_signal, components, signals
    
    def generate_bandwidth(self, k, f0, bandwidth):
        """
        Generate random bandwidths for k segments with overlaps.

        Args:
            k (int): Number of segments.
            f0 (float): Starting frequency of the signal (Hz).
            bandwidth (float): Total bandwidth of the signal (Hz).

        Returns:
            np.ndarray: A 2D array of shape (k, 2) where each row represents [start, end] frequencies of a segment.
        """

        # Step 1: Split the total bandwidth into k non-overlapping segments using Dirichlet
        alpha = np.ones(k)  # Adjust the concentration parameter for bias
        segment_bandwidths = np.random.dirichlet(alpha) * bandwidth


        # Step 2: Generate non-overlapping segment boundaries
        segment_frequencies = [f0]
        for bw in segment_bandwidths:
            segment_frequencies.append(segment_frequencies[-1] + bw)

        # Convert to a 2D array of [start, end] for each segment
        segments = np.array([[segment_frequencies[i], segment_frequencies[i + 1]] for i in range(k)])

        # Step 3: Adjust each segment to introduce overlaps
        for i in range(k - 1):  # No need to adjust the last segment
            # Stretch the segment by a random amount
            adjustment = np.random.normal(self.overlap_factor, self.overlap_std) * (segments[i+1, 1] - segments[i+1, 0])
            segments[i + 1, 0] = segments[i, 1]  # Ensure the next segment starts where the current one ends before stretching
            segments[i, 1] += adjustment  # Adjust the end of the current segment

        return segments
    

# Example usage
if __name__ == "__main__":
    # Define parameters
    fmin = 50  # Minimum frequency (Hz)
    fmax = 500  # Maximum frequency (Hz)
    duration = 1.0  # Duration of the signal (seconds)
    signal_types = ['sine']
    intermittence = 0.5  # Probability of generating an intermittent signal
    overlap_factor = 0.5  # Factor to determine the overlap between segments
    overlap_std = 0.1  # Standard deviation for the overlap adjustment

    # Create an instance of the generator
    generator = SyntheticSignalGenerator(fmin, fmax, duration, signal_types, intermittence, overlap_factor, overlap_std)

    # Generate a composite signal with k segments
    composite_signal, components, signals = generator.generate_signal(f0=generator.fmin, bandwidth=generator.fmax-generator.fmin, k=3)
    print("Signal 1 frequency:", signals[0].frequency)
    print("Signal 2 frequency:", signals[1].frequency)
    print("Signal 3 frequency:", signals[2].frequency)

