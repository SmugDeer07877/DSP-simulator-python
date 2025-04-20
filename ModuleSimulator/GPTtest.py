import numpy as np
from scipy.signal import upfirdn, fir_filter_design, firwin, convolve
from numpy import pi, sqrt, cos, sin
import matplotlib.pyplot as plt
from commpy.modulation import PSKModem

rolloff = 0.5
sps = 8
span_filter = 16
Rs = 1e6
fs = sps * Rs
Ts = 1 / fs
fc = 2.125e6
M = 4  # QPSK
k = int(np.log2(M))
span = 200
Ns = 8
T = Ns * Ts

modem = PSKModem(M)
data = np.random.randint(0, M, span)
b = modem.modulate(data) * np.exp(1j * pi/4)  # Apply pi/4 phase shift

from commpy.filters import rrcosfilter

gTX, t = rrcosfilter(N=span_filter * sps, alpha=rolloff, Ts=1/Rs, Fs=fs)
u_n = upfirdn(h=gTX, x=b, up=sps)
u_n = np.concatenate([u_n, np.zeros(7)])

# Assume gTX is defined
gRx = np.conj(gTX[::-1])  # Matched filter
r_filtered = convolve(u_n, gRx, mode='same')

# Adjust for filter delay
delay = (len(gTX) - 1) // 2
r_filtered = r_filtered[delay:-delay]  # remove transient edges

# Downsample
z = r_filtered[::sps]

# Decision: QPSK decoding (symbol closest to constellation)
# Define QPSK constellation (pi/4 rotated)
constellation = 1 / np.sqrt(2) * np.array([1+1j, -1+1j, -1-1j, 1-1j])
detected_symbols = [constellation[np.argmin(np.abs(s - constellation))] for s in z]

# Assuming b and b_estimate_final are numpy arrays of complex QPSK symbols
# wrong_real = np.sum(np.round(np.real(detected_symbols)) != np.round(np.real(b)))
# wrong_imag = np.sum(np.round(np.imag(detected_symbols)) != np.round(np.imag(b)))
#
# total_wrong_bits = wrong_real + wrong_imag
# BER = total_wrong_bits / (2 * len(b))  # 2 bits per QPSK symbol
# print(f"BER = {BER:.4f}")