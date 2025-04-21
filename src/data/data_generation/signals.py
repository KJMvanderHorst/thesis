import numpy as np

# Linear Amplitude Modulation
class LinearAMSignal:
    def __init__(self, b, a, fam, phi, duration, sample_rate):
        self.b = b
        self.a = a
        self.fam = fam
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        A_t = self.b * t + self.a
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)

# Sinusoidal Amplitude Modulation
class SinusoidalAMSignal:
    def __init__(self, fs, phi_s, fam, phi, duration, sample_rate):
        self.fs = fs
        self.phi_s = phi_s
        self.fam = fam
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        A_t = np.sin(2 * np.pi * self.fs * t + self.phi_s)
        return A_t * np.sin(2 * np.pi * self.fam * t + self.phi)

# Linear Frequency Modulation
class LinearFMSignal:
    def __init__(self, f0, B, T, phi, duration, sample_rate):
        self.f0 = f0
        self.B = B
        self.T = T
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        alpha = self.B / self.T
        return np.sin(2 * np.pi * (self.f0 + alpha * t) * t + self.phi)

# Sinusoidal Frequency Modulation
class SinusoidalFMSignal:
    def __init__(self, fc, fd, fm, phi, duration, sample_rate):
        self.fc = fc
        self.fd = fd
        self.fm = fm
        self.phi = phi
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        return np.sin(2 * np.pi * self.fc * t + self.fd / self.fm * np.sin(2 * np.pi * self.fm * t + self.phi))

# AM-FM Combination
class AMFMSignal:
    def __init__(self, am_signal, fm_signal):
        self.am_signal = am_signal
        self.fm_signal = fm_signal

    def generate(self):
        A_t = self.am_signal.generate()
        S_fm = self.fm_signal.generate()
        return A_t * S_fm

# Intermittent Signal
class IntermittentSignal:
    def __init__(self, base_signal, t0, tmax, duration, sample_rate):
        self.base_signal = base_signal
        self.t0 = t0
        self.tmax = tmax
        self.duration = duration
        self.sample_rate = sample_rate

    def rect(self, t):
        return np.where((t >= self.t0) & (t <= self.tmax), 1, 0)

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        return self.rect(t) * self.base_signal.generate()

# Stationary Sine Wave
class SineSignal:
    def __init__(self, frequency, amplitude, phase, duration, sample_rate):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duration = duration
        self.sample_rate = sample_rate

    def generate(self):
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate), endpoint=False)
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)