import numpy as np
from src.data.data_generation.signals import *


"""
composite_signal_generator.py

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
    def __init__(self, cfg):
        """
        Initialize the SyntheticSignalGenerator with parameters from the config.

        Args:
            cfg (DictConfig): Configuration object containing generator parameters.
        """
        # Configuration parameters
        self.cfg = cfg

        # Generator parameters
        self.fmin = cfg.generator.fmin
        self.fmax = cfg.generator.fmax
        self.duration = cfg.generator.duration
        self.signal_types = cfg.generator.signal_types
        self.intermittence = cfg.generator.intermittence
        self.overlap_factor = cfg.generator.overlap_factor
        self.overlap_std = cfg.generator.overlap_std

        # Constants
        self.min_modulation_frequency = cfg.constants.get("min_modulation_frequency", 0.1)
        self.max_modulation_frequency = cfg.constants.get("max_modulation_frequency", 10.0)
        self.min_amplitude = cfg.constants.get("min_amplitude", 0.5)
        self.max_amplitude = cfg.constants.get("max_amplitude", 1.5)
        self.intermittence_lower_bound = cfg.constants.get("intermittence_lower_bound", 0.1)
        self.intermittence_upper_bound = cfg.constants.get("intermittence_upper_bound", 0.9)
        self.max_frequency_deviation = cfg.constants.get("max_frequency_deviation", 5.0)
        self.sampling_rate = cfg.constants.get("sampling_rate", 1000)  # Default sample rate

    def generate_signal(self, f0, bandwidth, k):
        """
        Generate a synthetic signal of signals with various modulation types.
        The signals are generated based on the specified parameters such as
        frequency range, duration, and sample rate.

        Args:
            cfg (DictConfig): Configuration object containing generator parameters.
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
                duration = np.random.uniform(self.intermittence_lower_bound, self.intermittence_upper_bound) * self.duration
                t_start = np.random.uniform(0, self.duration - duration)  # Random start time for intermittent signal

            # Randomly select a signal type from the provided list
            signal_type = np.random.choice(self.signal_types)

            # Generate the signal dynamically
            signal = self.create_signal(signal_type, B_segments[i], duration)

            # Generate the signal and pad it to fit the total duration
            generated_signal = signal.generate()
            padded_signal = np.zeros(int(self.duration * self.sampling_rate))  # Create a zero array for the total duration
            t_start_idx = int(t_start * self.sampling_rate)
            t_end_idx = t_start_idx + len(generated_signal)
            padded_signal[t_start_idx:t_end_idx] = generated_signal  # Insert the intermittent signal

            signals.append(signal)
            components.append(padded_signal)

        # Sort B_segments by the start frequency
        sorted_indices = np.argsort(B_segments[:, 0])
        B_segments = B_segments[sorted_indices]

        # Reorder components to match the sorted B_segments
        components = [components[i] for i in sorted_indices]

        #this ensures that the components are in the same order as the frequency segments

        # Sum all components to create the composite signal
        composite_signal = np.sum(components, axis=0)
        return composite_signal, components, signals

    def create_signal(self, signal_type, bandwidth_segment, duration):
        """
        Factory method to create a signal based on its type and parameters.

        Args:
            signal_type (str): The type of signal to generate.
            bandwidth_segment (list): The [start, end] frequency range for the signal.
            duration (float): The duration of the signal.

        Returns:
            Signal: An instance of the appropriate signal class.
        """
        freq = np.random.uniform(bandwidth_segment[0], bandwidth_segment[1])
        starting_freq = np.random.uniform(0, 2 * np.pi)

        # Define a dictionary mapping signal types to their creation logic
        signal_generators = {
            'linear_am': lambda: LinearAMSignal(
                cfg=self.cfg,
                b=np.random.uniform(self.min_amplitude, self.max_amplitude),
                a=np.random.uniform(self.min_amplitude, self.max_amplitude),
                fam=freq,
                phi=starting_freq,
                duration=duration
            ),
            'sinusoidal_am': lambda: SinusoidalAMSignal(
                cfg=self.cfg,
                fs=np.random.uniform(self.min_modulation_frequency, self.max_modulation_frequency),
                phi_s=np.random.uniform(0, 2 * np.pi),
                fam=freq,
                phi=starting_freq,
                duration=duration
            ),
            'linear_fm': lambda: LinearFMSignal(
                cfg=self.cfg,
                f0=bandwidth_segment[0],
                B=bandwidth_segment[1] - bandwidth_segment[0],
                T=duration,
                phi=0
            ),
            'sinusoidal_fm': lambda: SinusoidalFMSignal(
                cfg=self.cfg,
                fc=freq,
                fd=np.random.uniform(0, self.max_frequency_deviation * freq),
                fm=np.random.uniform(self.min_modulation_frequency, self.max_modulation_frequency),
                phi=starting_freq,
                duration=duration
            ),
            'amfm': lambda: AMFMSignal(
                cfg=self.cfg,
                am_signal=LinearAMSignal(
                    b=np.random.uniform(self.min_amplitude, self.max_amplitude),
                    a=np.random.uniform(self.min_amplitude, self.max_amplitude),
                    fam=freq,
                    phi=starting_freq,
                    duration=duration
                ),
                fm_signal=LinearFMSignal(
                    cfg=self.cfg,
                    f0=bandwidth_segment[0],
                    B=bandwidth_segment[1] - bandwidth_segment[0],
                    T=duration,
                    phi=0
                )
            ),
            'sine': lambda: SineSignal(
                cfg=self.cfg,
                frequency=freq,
                phi=0,
                duration=duration
            )
        }

        # Use the dictionary to create the signal
        if signal_type in signal_generators:
            return signal_generators[signal_type]()
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

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
            adjustment = np.random.normal(self.overlap_factor, self.overlap_std) * (segments[i + 1, 1] - segments[i + 1, 0])
            segments[i + 1, 0] = segments[i, 1]  # Ensure the next segment starts where the current one ends before stretching
            segments[i, 1] += adjustment  # Adjust the end of the current segment

        return segments

