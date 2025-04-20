import math
import numpy as np


def modulator(data):
    modulated_data = []
    for i in range(0, len(data), 2):
        pair = str(data[i])+str(data[i+1])
        if pair == "00":
            modulated_data.append((1j+1)/math.sqrt(2))
        elif pair == "01":
            modulated_data.append((1j-1) / math.sqrt(2))
        elif pair == "10":
            modulated_data.append((-1j+1) / math.sqrt(2))
        elif pair == "11":
            modulated_data.append((-1j-1) / math.sqrt(2))
    return modulated_data



def demodulator (data):
    constellation = 1 / np.sqrt(2) * np.array([1+1j, -1+1j, -1-1j, 1-1j])
    demod_data = [constellation[np.argmin(np.abs(s - constellation))] for s in data]
    return demod_data



if __name__ == '__main__':
    data = [0, 0, 0, 1, 1, 0, 1, 1]
    modulated_data = modulator(data)
    print(modulated_data)
    demodulated_data = demodulator (modulated_data)
    print(demodulated_data)
