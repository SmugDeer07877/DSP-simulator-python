from commpy.filters import rrcosfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.signal import upfirdn

# Filter parameters
span = 20            # Filter length (must be odd)
alpha = 0.25               # Roll-off factor (0 < alpha ≤ 1)
symbol_rate = 1e6           # In symbols per second
sampling_rate = 3e6        # Number of samples per symbol
num_bits = int(1e5)

# Calculate sampling frequency
Ts = 1/symbol_rate
Fs = sampling_rate
sps = sampling_rate/symbol_rate
filter_len = span * sps + 1

#Ben's version:
#Fs = sps * symbol_rate

#Filter Delay:
delay = int((filter_len)-1// 2)

# Generate RRC filter
time_idx, sRRCfilter = rrcosfilter(int(filter_len), alpha, Ts, Fs)

# Matched filter is the same as the transmit filter
#matched_filter = taps

def rcosfilter(N, beta, Ts, Fs):
    t = (np.arange(N) - N / 2) / Fs
    return np.where(np.abs(2*t) == Ts / beta,
        np.pi / 4 * np.sinc(t/Ts),
        np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts) ** 2))

def plot_filter(filter):
    # Plot the filter
    plt.plot(filter)
    plt.title("Root Raised Cosine (RRC) Filter")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def upzero(data, sps):
    data_upzero = []
    for x in data:
        data_upzero.append(data[x])
        for i in range(int(sps-1)):
            data_upzero.append(0)
    return data_upzero

def upsample(data, sps):
    data_upsample = []
    for x in data:
        for i in range(int(sps)):
            data_upsample.append(x)
    return data_upsample

def zero_pad(data, delay):
    for i in range(delay):
        data.append(0)
    return data

def filter(filter, data, sps, delay):
    upsample_data = upzero(data,sps)
    filtered_data = np.convolve(filter, upsample_data, mode="same")
    #filtered = upfirdn(filter, data, int(sps))
    #filtered_data = np.concatenate(filtered_data, np.zeros(delay))
    filtered = zero_pad(filtered_data.tolist(), delay)
    return filtered

def matched_filter(data, filter):
    matched_filter = np.conj(filter[::-1])
    matched_data = np.convolve(matched_filter, data, mode="same")
    return matched_data

def unfilter(data, filter, sps):
    #downsample = data[::int(sps)]
    #fliped = np.flip(downsample)
    #delayed_data = data[delay : -delay]
    delay = (len(filter)-1)//2
    data = data[delay:-delay]
    #filtered = data[::sps]
    bits = (data > 0).astype(int)
    return bits

if __name__ == '__main__':
    #Create filter:
    #sRRCfilter = rcosfilter(span*sps, alpha, Ts, Fs)
    plot_filter(sRRCfilter)
    #Generate random bits and BPSK modulate (0 -> -1, 1 -> +1)
    data = np.random.randint(0, 2, num_bits)
    bpsk_symbols = 2 * data - 1  # BPSK modulation
    print(f"{len(bpsk_symbols)}")

    # plt.scatter(np.real(bpsk_symbols),np.imag(bpsk_symbols))
    # plt.show()

    #Filter data
    filtered_data = filter(sRRCfilter, bpsk_symbols, sps, delay)
    print(f"{len(filtered_data)}")
    # plt.scatter(np.real(filtered_data), np.imag(filtered_data))
    # plt.show()

    #Matched filter
    matched_data = matched_filter(filtered_data, sRRCfilter)
    print(f"{len(matched_data)}")


    # unfiltered_data = unfilter(matched_data, sps)
    # print(f"{len(unfiltered_data)}")

    #Test:
    print(f"{data[:10]}")
    # print(f"{unfiltered_data[:10]}")

    data1 = matched_data[::3]
    plt.scatter(np.real(data1), np.imag(data1))
    plt.show()
    fdata1 = unfilter(data1, sRRCfilter, sps)
    print(f"{len(data1)}")
    print("data1:", fdata1[:10])
    data2 = matched_data[1::3]
    plt.scatter(np.real(data2), np.imag(data2))
    plt.show()
    fdata2 = unfilter(data2, sRRCfilter, sps)
    print("data2:", fdata2[:10])
    data3 = matched_data[2::3]
    plt.scatter(np.real(data3), np.imag(data3))
    plt.show()
    fdata3 = unfilter(data3, sRRCfilter, sps)
    print("data3:", fdata3[:10])

    # 8. Calculate Bit Error Rate (BER)
    bit_errors = np.sum(data != fdata1)
    ber = bit_errors / num_bits
    print(f'Bit Error Rate: {ber:.4f} ({bit_errors} bit errors)')
    bit_errors = np.sum(data != fdata2)
    ber = bit_errors / num_bits
    print(f'Bit Error Rate: {ber:.4f} ({bit_errors} bit errors)')
    bit_errors = np.sum(data != fdata3)
    ber = bit_errors / num_bits
    print(f'Bit Error Rate: {ber:.4f} ({bit_errors} bit errors)')
    # plt.plot(np.abs(np.fft.fftshift(np.fft.fft(matched_data))))
    # plt.show()
