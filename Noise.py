from commpy.channels import awgn
import numpy as np
import matplotlib.pyplot as plt

def awgn_noise (signal, snr, rate):
    noisy_signal = awgn(signal, snr, rate)
    return noisy_signal


#sigma - controls the strength of the phase deviation
def phase_noise(signal, sigma):
    A = sigma * np.random.randn(len(signal))
    fftsignal = np.fft.fft(signal)
    fftsignal_noise = fftsignal * np.exp(1j * A)
    signal_noise = np.fft.ifft(fftsignal_noise)
    return signal_noise

def IQ_imbalance (signal, gain_db, phase_deg):
    gain = 10**(gain_db / 20)
    phase = np.deg2rad(phase_deg)
    I = np.real(signal)
    Q = np.imag(signal)
    I_imbalance = I
    Q_imbalance = gain * (np.cos(phase)*Q + np.sin(phase)*I)
    return I_imbalance + Q_imbalance*1j

def minor_sat(signal, a):
    if signal.type() == complex():
        return np.tanh(np.real(signal)*a) + 1j* np.tanh(np.imag(signal)*a)
    else:
        return np.tanh(signal*a)

def AM_noise(signal, cycles, min, max, p):
    '''
    parameters:
    signal: input signal
    cycles: normalized frequency (number of cycles over the full signal)
    min, max = gain fluctuation range
    p = intensity of amplitude variation (in percent [%])
    '''
    n = len(signal)
    noise = 1 + p * np.cos(2 * np.pi * cycles * np.arange(n) / n) * np.random.uniform(min, max, size = n)
    return noise * signal



def show_noise (t, signal, noise_signal, noisetype = None):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, noise_signal.real, label=f'With {noisetype}', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(noisetype)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_noisefft(signal, noise_signal, noisetype = None):
    plt.figure(figsize=(10, 4))
    plt.plot( np.fft.fft(signal), label='Original Signal')
    plt.plot( np.fft.fft(noise_signal), label='Original Signal')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title(noisetype)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    t_min = -1
    t_max = 1
    num_t = 1000
    t = np.linspace(t_min, t_max, num_t)
    f = 4
    signal = np.cos(2 * np.pi * f * t)
    #noise_signal = np.cos(2 * np.pi * f * t + sigma * np.random.randn(len(t)))
    #noise_signal = phase_noise(signal)
    #noise_signal = awgn_noise(signal)
    noise_signal = minor_sat(signal, 3)
    show_noise(t, signal, noise_signal, "awgn Noises")
    show_noisefft(signal, noise_signal, "awgn Noises")


