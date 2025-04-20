import numpy as np
import matplotlib.pyplot as plt


def ber (data, new_data):
    errors = sum(b1 != b2 for b1, b2 in zip(data, new_data))
    ber = errors / len(data)
    return ber


def snr(original_signal, received_signal):
    original_signal = np.array(original_signal)
    received_signal = np.array(received_signal)
    noise = received_signal - original_signal

    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def plot_bersnr(ber, snr):
    plt.scatter(snr, ber)
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.grid()
    plt.show()

def plot_constelation(og_signal, new_signal):
    plt.scatter(np.real(new_signal), np.imag(new_signal))
    plt.scatter(np.real(og_signal), np.imag(og_signal))
    plt.grid()
    plt.show()


