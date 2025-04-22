import numpy as np
from signals import *
import os
import json
from data_config import INTERMITTENCE_LOWER_BOUND
from data_config import INTERMITTENCE_UPPER_BOUND


"""
generate_dataset.py

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
    generate_and_store_dataset: A function to generate a dataset of composite signals
        and store it as a CSV file along with the parameters used.
"""

class SyntheticSignalGenerator:
    """
    A class to generate synthetic a dataset of signals with various modulation types.
    Attributes:
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        duration (float): Duration of the signal in seconds.
        signal_types (list): List of signal types to generate. Options include:
            'linear_am', 'sinusoidal_am', 'linear_fm', 'sinusoidal_fm',
            'amfm' and 'sine'.
    """
    def __init__(self, fmin, fmax, duration, signal_types, intermittence):
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.signal_types = signal_types
        self.intermittence = intermittence

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
        B_segments = np.linspace(self.fmin, self.fmax, k + 1)
        components = []

        for i in range(k):
            f0 = (B_segments[i] + B_segments[i + 1]) / 2
            # Randomly select if a signal has intermittence
            duration = np.random.uniform(INTERMITTENCE_LOWER_BOUND, INTERMITTENCE_UPPER_BOUND) * self.duration if np.random.uniform(0, 1) > self.intermittence else self.duration

            # Randomly select a signal type from the provided list
            signal_type = np.random.choice(self.signal_types)

            match signal_type:
                case 'linear_am':
                    signal = LinearAMSignal(b=0.5, a=1, fam=f0, phi=0, duration = duration)
                case 'sinusoidal_am':
                    signal = SinusoidalAMSignal(fs=1, phi_s=0, fam=f0, phi=0, duration=duration)
                case 'linear_fm':
                    signal = LinearFMSignal(f0=f0, B=B_segments[i + 1] - B_segments[i], T=duration, phi=0)
                case 'sinusoidal_fm':
                    signal = SinusoidalFMSignal(fc=f0, fd=5, fm=2, phi=0, duration=duration)
                case 'amfm':
                    am = LinearAMSignal(b=0.5, a=1, fam=f0, phi=0, duration=self.duration)
                    fm = LinearFMSignal(f0=f0, B=5, T=self.duration, phi=0, duration=duration)
                    signal = AMFMSignal(am_signal=am, fm_signal=fm)
                case _:
                    signal = SineSignal(frequency=f0, amplitude=1, phase=0, duration=duration)

            components.append(signal.generate())

        composite_signal = np.sum(components, axis=0)
        return composite_signal, components

def generate_and_store_dataset(generator, num_signals, k):
    """
    Generate a dataset of composite signals and store it as an NPZ file along with the parameters used.

    Args:
        generator (SyntheticSignalGenerator): An instance of the SyntheticSignalGenerator class.
        num_signals (int): Number of composite signals to generate.
        k (int): Number of frequency segments for each signal.

    Returns:
        None
    """
    # Ensure the output folder exists
    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a list to store the dataset
    dataset = []

    for _ in range(num_signals):
        # Generate a composite signal and its components
        composite_signal, _ = generator.generate_signal(k)
        dataset.append(composite_signal)

    # Convert the dataset to a NumPy array
    dataset = np.array(dataset)

    # Save the dataset as an NPZ file
    dataset_path = os.path.join(output_folder, "composite_signals.npz")
    np.savez_compressed(dataset_path, dataset=dataset)

    # Save the parameters used for the entire dataset as a JSON file
    params = {
        "fmin": generator.fmin,
        "fmax": generator.fmax,
        "duration": generator.duration,
        "signal_types": generator.signal_types,
        "intermittence": generator.intermittence,
        "num_signals": num_signals,
        "num_segments": k
    }
    params_path = os.path.join(output_folder, "parameters.json")
    with open(params_path, "w") as params_file:
        json.dump(params, params_file, indent=4)

    print(f"Dataset saved to {dataset_path}")
    print(f"Parameters saved to {params_path}")


