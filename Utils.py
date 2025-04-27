import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift


def save_plot(subfolder_name, filename):
    """
       Save a matplotlib figure into a specific subfolder inside a parent folder.

       Args:
       fig (matplotlib.figure.Figure): The figure to save.
       parent_folder (str): Path to the existing parent folder.
       subfolder_name (str): Name of the subfolder to create inside the parent folder.
       filename (str): Name of the file to save (include file extension, e.g., 'figure.png').
       dpi (int, optional): Resolution of the saved figure. Default is 300.
    """
    parent_folder = r"C:\Users\adibl\PycharmProjects\Figures"
    save_path = os.path.join(parent_folder, subfolder_name)
    os.makedirs(save_path, exist_ok=True)
    full_filename = os.path.join(save_path, filename)
    plt.savefig(full_filename)
    print(f"Figure saved to: {full_filename}")


def plot_time (signal, noise_signal, file_name):
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
    plt.plot(torch.real(signal), label='Lab Signal (I)', color = 'DarkRed')
    plt.plot(torch.real(noise_signal), label='Real Signal (I)', color = "DarkOrange", alpha = 0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Time Plot I")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(file_name, "Time_I")
    plt.show()
    plt.figure(figsize=(10, 4))
    plt.plot(torch.imag(signal), label='Lab Signal (Q)', color = 'DarkBlue')
    plt.plot(torch.imag(noise_signal), label='Real Signal (Q)', color = "Cyan", alpha = 0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Time Plot Q")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(file_name, "Time_Q")
    plt.show()

def plot_fft(signal, noise_signal, file_name):
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
    plt.plot( 10*np.log10(np.abs(fftshift(fft(noise_signal)))), label='Real Signal', alpha = 0.7)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title("FFT Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(file_name, "fft")
    plt.show()


def plot_constelation(signal, noise_signal, file_name):
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
    plt.scatter(signal.real, signal.imag, label = "Lab Signal")
    plt.scatter(noise_signal.real, noise_signal.imag, label = "Real Signal", alpha = 0.5)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title("Constellation Plot")
    plt.grid()
    save_plot(file_name, "constellation")
    plt.show()

def show_signal(signal, noise_signal, file_name):
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
    plot_time(signal, noise_signal, file_name)
    plot_fft(signal, noise_signal, file_name)
    plot_constelation(signal, noise_signal, file_name)