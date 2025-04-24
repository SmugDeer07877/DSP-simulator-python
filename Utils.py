import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift


def plot_time (signal, noise_signal, noisetype = None):
    '''
    Plots real and imag components of original signal and noisy signal in time domain
    -------
    Input:
    signal = 1D PyTorch tensor
    Complex signal
    noise_signal = 1D PyTorch tensor
    Complex signal with added noise
    noisetype = str
    Noise type added to noise_signal
    '''
    plt.figure(figsize=(10, 4))
    # plt.plot(signal, label='Original Signal')
    plt.plot(torch.real(signal), label='Lab Signal (I)')
    plt.plot(torch.real(noise_signal), label='Real Signal (I)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(noisetype)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 4))
    # plt.plot(signal, label='Original Signal')
    plt.plot(torch.imag(signal), label='Lab Signal (Q)')
    plt.plot(torch.imag(noise_signal), label='Real Signal (Q)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(noisetype)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fft(signal, noise_signal, noisetype = None):
    '''
    Plots the original signal and noisy signal in frequency domain
    -------
    Input:
    signal = 1D PyTorch tensor
    Complex signal
    noise_signal = 1D PyTorch tensor
    Complex signal with added noise
    noisetype = str
    Noise type added to noise_signal
    '''
    plt.figure(figsize=(10, 4))
    plt.plot( 10*np.log10(np.abs(fftshift(fft(signal)))), label='Lab Signal')
    plt.plot( 10*np.log10(np.abs(fftshift(fft(noise_signal)))), label='Real Signal')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title(noisetype)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_constelation(signal, noise_signal, noisetype = None):
    '''
    Plots original signal and noisy signal in constellation plot
    -------
    Input:
    signal = 1D PyTorch tensor
    Complex signal
    noise_signal = 1D PyTorch tensor
    Complex signal with added noise
    noisetype = str
    Noise type added to noise_signal
    '''
    plt.scatter(signal.real, signal.imag)
    plt.scatter(noise_signal.real, noise_signal.imag)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title(noisetype)
    plt.grid()
    plt.show()

def show_signal(signal, noise_signal, noisetype = None):
    '''
    Plots original signal and noisy signal in time, frequency and constelation plots
    -------
    Input:
    signal = 1D PyTorch tensor
    Complex signal
    noise_signal = 1D PyTorch tensor
    Complex signal with added noise
    noisetype = str
    Noise type added to noise_signal
    '''
    plot_time(signal, noise_signal, noisetype = None)
    plot_fft(signal, noise_signal, noisetype = None)
    plot_constelation(signal, noise_signal, noisetype = None)