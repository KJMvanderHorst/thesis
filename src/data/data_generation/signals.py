"""
signals.py

This module provides classes for generating various types of signals, including
Amplitude Modulation (AM), Frequency Modulation (FM), and combinations of these.
It also includes support for intermittent signals and stationary sine waves.

Classes:
    LinearAMSignal: Generates a linear amplitude-modulated signal.
    SinusoidalAMSignal: Generates a sinusoidal amplitude-modulated signal.
    LinearFMSignal: Generates a linear frequency-modulated signal.
    SinusoidalFMSignal: Generates a sinusoidal frequency-modulated signal.
    AMFMSignal: Combines AM and FM signals.
    IntermittentSignal: Generates an intermittent signal based on a base signal.
    SineSignal: Generates a stationary sine wave signal.
"""

import numpy as np

class LinearAMSignal:
    """
    Generates a linear amplitude-modulated (AM) signal.

    Args:
        b (float): Maximum amplitude.
        a (float): Minimum amplitude.
        fam (float): Frequency of the carrier signal (Hz).
        phi (float): Phase of the carrier signal (radians).
        duration (float): Duration of the signal (seconds).
        sample_rate (float): Sampling rate (samples per second).
    """

    def __init__(self, b, a, fam, phi, duration, sample_rate):
        self.b = b
        self.a = a
        self.fam = fam
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        """
        Generates the linear AM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        A_t = self.a / self.b * t + self.a
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)


class SinusoidalAMSignal:
    """
    Generates a sinusoidal amplitude-modulated (AM) signal.

    Args:
        fs (float): Frequency of the amplitude modulation (Hz).
        phi_s (float): Phase of the amplitude modulation (radians).
        fam (float): Frequency of the carrier signal (Hz).
        phi (float): Phase of the carrier signal (radians).
        duration (float): Duration of the signal (seconds).
        sample_rate (float): Sampling rate (samples per second).
    """

    def __init__(self, fs, phi_s, fam, phi, duration, sample_rate):
        self.fs = fs
        self.phi_s = phi_s
        self.fam = fam
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        """
        Generates the sinusoidal AM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        A_t = np.sin(2 * np.pi * self.fs * t + self.phi_s)
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)


class LinearFMSignal:
    """
    Generates a linear frequency-modulated (FM) signal.

    Args:
        f0 (float): Initial frequency of the signal (Hz).
        B (float): Bandwidth of the frequency modulation (Hz).
        T (float): Time duration over which the frequency modulation occurs (seconds).
        phi (float): Phase of the signal (radians).
        duration (float): Duration of the signal (seconds).
        sample_rate (float): Sampling rate (samples per second).
    """

    def __init__(self, f0, B, T, phi, duration, sample_rate):
        self.f0 = f0
        self.B = B
        self.T = T
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        """
        Generates the linear FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        alpha = self.B / self.T
        return np.sin(2 * np.pi * (self.f0 + alpha * t) * t + self.phi)


class SinusoidalFMSignal:
    """
    Generates a sinusoidal frequency-modulated (FM) signal.

    Args:
        fc (float): Carrier frequency (Hz).
        fd (float): Frequency deviation (Hz).
        fm (float): Modulation frequency (Hz).
        phi (float): Phase of the signal (radians).
        duration (float): Duration of the signal (seconds).
        sample_rate (float): Sampling rate (samples per second).
    """

    def __init__(self, fc, fd, fm, phi, duration, sample_rate):
        self.fc = fc
        self.fd = fd
        self.fm = fm
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        """
        Generates the sinusoidal FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        return np.sin(2 * np.pi * self.fc * t + self.fd / self.fm * np.sin(2 * np.pi * self.fm * t + self.phi))


class AMFMSignal:
    """
    Combines an amplitude-modulated (AM) signal and a frequency-modulated (FM) signal.

    Args:
        am_signal (LinearAMSignal or SinusoidalAMSignal): The AM signal object.
        fm_signal (LinearFMSignal or SinusoidalFMSignal): The FM signal object.
    """

    def __init__(self, am_signal, fm_signal):
        self.am_signal = am_signal
        self.fm_signal = fm_signal

    def generate(self):
        """
        Generates the combined AM-FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        A_t = self.am_signal.generate()
        S_fm = self.fm_signal.generate()
        return A_t * S_fm


class SineSignal:
    """
    Generates a stationary sine wave signal.

    Args:
        frequency (float): Frequency of the sine wave (Hz).
        amplitude (float): Amplitude of the sine wave.
        phase (float): Phase of the sine wave (radians).
        duration (float): Duration of the signal (seconds).
        sample_rate (float): Sampling rate (samples per second).
    """

    def __init__(self, frequency, amplitude, phase, duration, sample_rate):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        """
        Generates the sine wave signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)