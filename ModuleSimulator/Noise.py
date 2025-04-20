from commpy.channels import awgn
import numpy as np
import matplotlib.pyplot as plt

def awgn_noise (signal, snr, rate):
    noisy_signal = awgn(signal, snr, rate)
    return noisy_signal


sigma = 5 # controlls the strength of the phase deviation
def phase_noise(signal):
    A = sigma * np.random.randn(len(signal))
    fftsignal = np.fft.fft(signal)
    fftsignal_noise = fftsignal * np.exp(1j * A)
    signal_noise = np.fft.ifft(fftsignal_noise)
    return signal_noise

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
    noise_signal = phase_noise(signal)
    #noise_signal = awgn_noise(signal)
    show_noise(t, signal, noise_signal, "Phase Noises")
    show_noisefft(signal, noise_signal, "Phase Noises")


