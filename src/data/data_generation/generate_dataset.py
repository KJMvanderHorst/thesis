"""
generate_dataset.py

This module provides a function to generate a synthetic dataset of signals
with various modulation types. The signals are generated based on specified
parameters such as frequency range, duration, and sample rate. The generated
signals can be used for testing and training machine learning models in
signal processing applications.

Classes:

"""

import numpy as np
from signals import *

class SyntheticSignalGenerator:
    """
    A class to generate synthetic signals with various modulation types.
    Attributes:
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sample rate for the signal generation.
    """
    def __init__(self, fmin, fmax, duration, sample_rate):
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self, k ,signal_types):
        """
        Generate a synthetic signal of signals with various modulation types.
        The signals are generated based on the specified parameters such as
        frequency range, duration, and sample rate.
        Args:
            k (int): Number of frequency segments.
            signal_types (list): List of signal types to generate. Options include:
                'linear_am', 'sinusoidal_am', 'linear_fm', 'sinusoidal_fm',
                'amfm' and 'sine'.
        Returns:
            composite_signal (numpy.ndarray): The generated composite signal.
            components (list): A list of individual signal components.
        """
        # Generate frequency segments
        B_segments = np.linspace(self.fmin, self.fmax, self.k + 1)
        components = []

        for i in range(k):
            f0 = (B_segments[i] + B_segments[i + 1]) / 2
            signal_type = np.random.choice(signal_types)
            # Randomly select a signal type from the provided list

            if signal_type == 'linear_am':
                signal = LinearAMSignal(b=0.5, a=1, fam=f0, phi=0, duration=self.duration, sample_rate=self.sample_rate)
            elif signal_type == 'sinusoidal_am':
                signal = SinusoidalAMSignal(fs=1, phi_s=0, fam=f0, phi=0, duration=self.duration, sample_rate=self.sample_rate)
            elif signal_type == 'linear_fm':
                signal = LinearFMSignal(f0=f0, B=B_segments[i + 1] - B_segments[i], T=self.duration, phi=0, duration=self.duration, sample_rate=self.sample_rate)
            elif signal_type == 'sinusoidal_fm':
                signal = SinusoidalFMSignal(fc=f0, fd=5, fm=2, phi=0, duration=self.duration, sample_rate=self.sample_rate)
            elif signal_type == 'amfm':
                am = LinearAMSignal(b=0.5, a=1, fam=f0, phi=0, duration=self.duration, sample_rate=self.sample_rate)
                fm = LinearFMSignal(f0=f0, B=5, T=self.duration, phi=0, duration=self.duration, sample_rate=self.sample_rate)
                signal = AMFMSignal(am_signal=am, fm_signal=fm)
            else:
                signal = SineSignal(frequency=f0, amplitude=1, phase=0, duration=self.duration, sample_rate=self.sample_rate)

            components.append(signal.generate())

        composite_signal = np.sum(components, axis=0)
        return composite_signal, components
    
