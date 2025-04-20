import numpy as np
import matplotlib.pyplot as plt
import scipy
'''
This page is an explanation on the implementation of DSP in python
'''
#Step (1) - Generating a signal
#Creating the timeline
t_min = -1
t_max = 1
num_t = 1000
t = np.linspace(t_min, t_max, num_t)

#Creating the signal function
f = 2
xt = np.cos(2*np.pi*f*t)

#Ploting the signal
plt.plot(t,xt)
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("Plot of cos(2*pi*f*t)")
plt.show()

#Step (2) - Sampling
# Define sampling frequency (should be at least twice the frequency) and sample time
Fs = 5
Ts = 1/Fs
# Create sampling pulses
sample_pulse = np.arange(t_min, t_max, Ts)
# Plot Sampling Pulses
plt.stem(sample_pulse, np.ones(len(sample_pulse)))
plt.show()
# Show the values of the samples
xt_sampled = np.cos(2*np.pi*f*sample_pulse)
plt.stem(sample_pulse, xt_sampled)
# Show the signal too
plt.plot(t, xt, 'r')
plt.show()

# Step (3) - Recovering signal
x_rs, t_rs = scipy.signal.resample(xt_sampled, 1000, sample_pulse)
plt.plot(t_rs, x_rs)
plt.show()



