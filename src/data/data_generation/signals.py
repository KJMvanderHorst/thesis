import numpy as np

"""
signals.py

This module provides classes for generating various types of signals, including
Amplitude Modulation (AM), Frequency Modulation (FM), and combinations of these.
It also includes support for stationary sine waves.

Classes:
    BaseSignal: Base class for all signal types, handling common parameters like
        sampling rate, minimum amplitude, and maximum amplitude.
    LinearAMSignal: Generates a linear amplitude-modulated signal.
    SinusoidalAMSignal: Generates a sinusoidal amplitude-modulated signal.
    LinearFMSignal: Generates a linear frequency-modulated signal.
    SinusoidalFMSignal: Generates a sinusoidal frequency-modulated signal.
    AMFMSignal: Combines AM and FM signals.
    SineSignal: Generates a stationary sine wave signal.
"""

class BaseSignal:
    """
    Base class for all signal types. Handles common parameters like sampling rate,
    minimum amplitude, and maximum amplitude.

    Attributes:
        sampling_rate (float): Sampling rate (samples per second).
        min_amplitude (float): Minimum amplitude of the signal.
        max_amplitude (float): Maximum amplitude of the signal.
    """

    def __init__(self, cfg, duration, phi = 0):
        """
        Initialize the base signal with common parameters.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            duration (float): Duration of the signal (seconds).
            phi (float): Starting phase of the signal (radians).
        """
        self.sampling_rate = cfg.constants.get("sampling_rate", 1000)
        self.min_amplitude = cfg.constants.get("min_amplitude", 0.5)
        self.max_amplitude = cfg.constants.get("max_amplitude", 1.5)
        self.duration = duration
        self.phi = phi


class LinearAMSignal(BaseSignal):
    """
    Generates a linear amplitude-modulated (AM) signal.

    Attributes:
        b (float): Maximum amplitude.
        a (float): Minimum amplitude.
        fam (float): Frequency of the carrier signal (Hz).
        phi (float): Phase of the carrier signal (radians).
        duration (float): Duration of the signal (seconds).
    """

    def __init__(self, cfg, b, a, fam, phi, duration):
        """
        Initialize the LinearAMSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            b (float): Maximum amplitude.
            a (float): Minimum amplitude.
            fam (float): Frequency of the carrier signal (Hz).
        """
        super().__init__(cfg, duration, phi)
        self.b = b
        self.a = a
        self.fam = fam

    def generate(self):
        """
        Generates the linear AM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
        A_t = self.a / self.b * t + self.a
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)


class SinusoidalAMSignal(BaseSignal):
    """
    Generates a sinusoidal amplitude-modulated (AM) signal.

    Attributes:
        fs (float): Frequency of the amplitude modulation (Hz).
        phi_s (float): Phase of the amplitude modulation (radians).
        fam (float): Frequency of the carrier signal (Hz).
        phi (float): Phase of the carrier signal (radians).
        duration (float): Duration of the signal (seconds).
    """

    def __init__(self, cfg, fs, phi_s, fam, duration, phi, a, b):

        """
        Initialize the SinusoidalAMSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            fs (float): Frequency of the amplitude modulation (Hz).
            phi_s (float): Phase of the amplitude modulation (radians).
            fam (float): Frequency of the carrier signal (Hz).
            duration (float): Duration of the signal (seconds).
            phi (float): Phase of the carrier signal (radians).
            b (float): Maximum amplitude.
            a (float): Minimum amplitude.

        """
        super().__init__(cfg, duration, phi)
        self.fs = fs
        self.phi_s = phi_s
        self.fam = fam

        self.a = a
        self.b = b


    def generate(self):
        """
        Generates the sinusoidal AM signal.

        Returns:
            np.ndarray: The generated signal.
        """

        mod_index = (self.b - self.a) / (self.b + self.a)
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
        A_t = (1 + mod_index * np.sin(2 * np.pi * self.fs * t + self.phi_s))* self.a  # Envelope is always ≥ 0
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)


class LinearFMSignal(BaseSignal):
    """
    Generates a linear frequency-modulated (FM) signal.

    Attributes:
        f0 (float): Initial frequency of the signal (Hz).
        B (float): Bandwidth of the frequency modulation (Hz).
        T (float): Time duration over which the frequency modulation occurs (seconds).
        phi (float): Phase of the signal (radians).
        duration (float): Duration of the signal (seconds).
    """

    def __init__(self, cfg, f0, B, T, phi, duration):
        """
        Initialize the LinearFMSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            f0 (float): Initial frequency of the signal (Hz).
            B (float): Bandwidth of the frequency modulation (Hz).
            T (float): Time duration over which the frequency modulation occurs (seconds).
            phi (float): Phase of the signal (radians).
            duration (float): Duration of the signal (seconds).
        """
        super().__init__(cfg, duration, phi)
        self.f0 = f0
        self.B = B
        self.T = T

    def generate(self):
        """
        Generates the linear FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
        alpha = self.B / self.T
        A = np.random.uniform(self.min_amplitude, self.max_amplitude)
        return A * np.sin(2 * np.pi * (self.f0 + alpha * t) * t + self.phi)


class SinusoidalFMSignal(BaseSignal):
    """
    Generates a sinusoidal frequency-modulated (FM) signal.

    Attributes:
        fc (float): Carrier frequency (Hz).
        fd (float): Frequency deviation (Hz).
        fm (float): Modulation frequency (Hz).
        phi (float): Phase of the signal (radians).
        duration (float): Duration of the signal (seconds).
    """

    def __init__(self, cfg, fc, fd, fm, phi, duration):
        """
        Initialize the SinusoidalFMSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            fc (float): Carrier frequency (Hz).
            fd (float): Frequency deviation (Hz).
            fm (float): Modulation frequency (Hz).
            phi (float): Phase of the signal (radians).
            duration (float): Duration of the signal (seconds).
        """
        super().__init__(cfg, duration, phi)
        self.fc = fc
        self.fd = fd
        self.fm = fm

    def generate(self):
        """
        Generates the sinusoidal FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
        A = np.random.uniform(self.min_amplitude, self.max_amplitude)

        return A * np.sin(2 * np.pi * self.fc * t + self.fd / self.fm * np.sin(2 * np.pi * self.fm * t))


class AMFMSignal(BaseSignal):
    """
    Combines an amplitude-modulated (AM) signal and a frequency-modulated (FM) signal.

    Attributes:

        fm_signal (BaseSignal): The FM signal object.
        am_signal (BaseSignal): The AM signal object.
    """

    def __init__(self, cfg,  am_type, fm_signal, am_signal):

        """
        Initialize the AMFMSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            fm_signal (BaseSignal): The FM signal object.
            am_signal (BaseSignal): The AM signal object.
            am_type (str): Type of AM signal ('linear' or 'sinusoidal').

        """
        super().__init__(cfg, fm_signal.duration)
        self.am_type = am_type

        self.fm_signal = fm_signal
        self.am_signal = am_signal

    def generate(self):
        """
        Generates the combined AM-FM signal.

        Returns:
            np.ndarray: The generated signal.
        """
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)

        #generate the AM envelope
        if self.am_type == 'sinusoidal':
            mod_index = (self.am_signal.b - self.am_signal.a) / (self.am_signal.b + self.am_signal.a)
            A_t = (1 + mod_index * np.sin(2 * np.pi * self.am_signal.fs * t + self.am_signal.phi_s))* self.am_signal.a  # Envelope is always ≥ 0
        elif self.am_type == 'linear':
            A_t = self.am_signal.a / self.am_signal.b * t + self.am_signal.a
        else:
            raise ValueError("Invalid AM type. Choose 'linear' or 'sinusoidal'.")
        
        #generate the FM signal
        fm_signal = self.fm_signal.generate()
        return A_t * fm_signal



class SineSignal(BaseSignal):
    """
    Generates a stationary sine wave signal.

    Attributes:
        frequency (float): Frequency of the sine wave (Hz).
        phase (float): Phase of the sine wave (radians).
        duration (float): Duration of the signal (seconds).
    """

    def __init__(self, cfg, frequency, phi, duration):
        """
        Initialize the SineSignal.

        Args:
            cfg (DictConfig): Configuration object containing constants.
            frequency (float): Frequency of the sine wave (Hz).
            phase (float): Phase of the sine wave (radians).
            duration (float): Duration of the signal (seconds).
        """
        super().__init__(cfg, duration, phi)
        self.frequency = frequency

    def generate(self):
        """
        Generates the sine wave signal.

        Returns:
            np.ndarray: The generated signal.
        """
        amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
        t = np.linspace(0, self.duration, int(self.duration * self.sampling_rate), endpoint=False)
        return amplitude * np.sin(2 * np.pi * self.frequency * t + self.phi)