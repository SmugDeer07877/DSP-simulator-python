#Unfinished Work:
#hamming - make the divisible by 11 code into a function called hamming.
# call other functions encode and decode
# QPSK De-modulator - add a function to distinguish the incoming signals into the correct symbols

from read_image import image_to_bits
from Hamming import hamming, dehamming
from Interleaver import interleaver, deinterleaver
from Modulator import modulator, demodulator
from Filter import fillter, matched_fillter
from Noise import awgn_noise
from write_image import bits_to_image
from Plot import ber, snr, plot_bersnr, plot_constelation


#Encoding:
filepath = input("input picture filepath:")
data = image_to_bits(filepath)
print(len(data))

#Hamming:
if len(data) % 11 != 0:
    for l in range(11-(len(data)%11)):
        data.append(0)
print(len(data)/11)
hamming_data = []
for n in range(len(data)//11):
    packet = []
    for i in range(11):
        packet.append(data[n*11+i])
    hamming_data.extend(hamming(packet))

#Interleaving:
interleaved_data = interleaver(hamming_data)

#QPSK Modulation:
modulated_data = modulator(interleaved_data)

#sRRC Filter:
#filltered_data = fillter(modulated_data)

#Loss simulation:
noise_data = awgn_noise(modulated_data, 10, 1)

#Matched Filter:
#matched_data = matched_fillter(filltered_data)

#QPSK demodulation:
demodulated_data = demodulator(noise_data)

#De-Interleaver:
deinterleaved_data = deinterleaver(demodulated_data)

#De-Hamming:
dehamming_data = []
for l in range(int(len(deinterleaved_data)/16)):
    pocket = []
    for m in range(16):
        pocket.append(deinterleaved_data[l*16+m])
    dehamming_data.extend(dehamming(pocket))

#Reconstract Image:
bits_to_image(dehamming_data)

#BER/SNR Plot
print(ber(data, dehamming_data))
print(snr(modulated_data, noise_data))
plot_constelation(modulated_data, noise_data)


