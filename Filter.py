import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from IPython import embed
N = 2
Ts = 2e-6
Sample_R = 15e6

def plot_signal(x, title=None):
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(x))))
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def fillter(data):
    os = Sample_R
    sRRCfillter = rrcosfilter(N, alpha=0.2, Ts=Ts, Fs=os)[1]
    # plot_signal(sRRCfillter, "srrcfilter")
    filtered_data = np.convolve(data, sRRCfillter, mode="valid")
    return filtered_data.tolist()

def matched_fillter(data):
    os = Sample_R
    sRRCfillter = rrcosfilter(N, alpha=0.2, Ts=Ts, Fs=os)[1]
    data = np.flip(data)
    plot_signal(sRRCfillter, "srrc matched filter")

    matched_data = np.convolve(data,sRRCfillter, mode="valid")
    return matched_data.tolist()

def placeholder_filter(data):
    filtered_data = np.convolve(data, 1, mode="same")
    return filtered_data



if __name__ == '__main__':
    data = np.random.choice([-1, 1], size=int(1e5))
    data_og = data
    data_upsample = []
    for x in data:
        data_upsample.append(x)
        data_upsample.append(x)
        data_upsample.append(x)

    data = data_upsample

    plot_signal(data, "Upsampled data")
    print(data[:10])
    print("len(data):", len(data))
    filltered_data = fillter(data)
    print("len(filtered_data):", len(filltered_data))
    plot_signal(filltered_data, "Filterd Data")
    matched = matched_fillter(filltered_data)
    print("len(matched):", len(matched))
    print("matched:", matched[:20])
    data_rect = data_upsample[:len(matched)]
    plot_signal(matched, "Matched filter data")
    matched_class = [1 if x > 0 else -1 for x in matched]
    print(f"{len(matched_class)=}")
    print("matched_class:", matched_class[:20])
    plot_signal(matched_class, "matched class")
    data_rec1 = matched_class[::3]
    data_rec2 = matched_class[1::3]
    data_rec3 = matched_class[2::3]

    N_l = 100
    plt.plot(data_og[:N_l], label="data og")
    plt.plot(data_rec1[:N_l], label="data rec")
    plt.title("data vs data_rec1 time")
    plt.legend()
    plt.show()

    orig_rect_conv1 = np.convolve(data_rec1, data_rect, mode="full")
    orig_rect_conv2 = np.convolve(data_rec2, data_rect, mode="full")
    orig_rect_conv3 = np.convolve(data_rec3, data_rect, mode="full")

    plt.plot(orig_rect_conv1)
    print(np.argmax(orig_rect_conv1))
    plt.show()
    plt.plot(orig_rect_conv2)
    print(np.argmax(orig_rect_conv2))
    plt.show()
    plt.plot(orig_rect_conv3)
    print(np.argmax(orig_rect_conv3))
    plt.show()
    # embed()
    data_rect= data_og[:len(data_rec1)]
    error_signal = [1 if x != y else 0 for x,y in zip(data_rec1, data_rect)]
    error1 = sum(error_signal)/len(error_signal)
    print(f"{error1=}")

    error_signal = [1 if x != y else 0 for x,y in zip(data_rec2, data_rect)]
    error2 = sum(error_signal)/len(error_signal)
    print(f"{error2=}")

    error_signal = [1 if x != y else 0 for x,y in zip(data_rec3, data_rect)]
    error3 = sum(error_signal)/len(error_signal)
    print(f"{error3=}")