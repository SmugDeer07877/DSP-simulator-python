import torch
import numpy as np
from commpy.channels import awgn
from typing import Tuple
from enum import Enum

from Utils import show_signal

class TUPLE_INDEX(Enum):
    MIN = 0
    MAX = 1

class Augmentation:
    def __init__(self, snr: Tuple, rate: Tuple, sigma: Tuple,
                 gain_db: Tuple, phase_deg: Tuple, a: Tuple,
                 cycles:Tuple, p:Tuple, noise:Tuple):
        self.snr_params = snr
        self.rate_params = rate
        self.sigma_params = sigma
        self.gain_params = gain_db
        self.phase_params = phase_deg
        self.a_params = a
        self.cycles_params = cycles
        self.p_params = p
        self.noise_params = noise


    def _awgn_noise (self, signal):
        '''
        Input:
        signal = 1D PyTorch tensor
        Complex signal
        -------
        Parameters:
        snr [float] = Output SNR required in dB.
        rate [float] = Rate of the a FEC code used if any, otherwise 1.
        -------
        Returns:
        noise_signal = 1D PyTorch tensor
        Output signal from the channel with the specified SNR.
        '''
        signal = signal.numpy()
        random_snr = np.random.randint(self.snr_params[TUPLE_INDEX.MIN.value] * 10, self.snr_params[TUPLE_INDEX.MAX.value] * 10) / 10
        random_rate = 1.0
        noisy_signal = awgn(signal,random_snr, random_rate)
        noise_signal = torch.from_numpy(noisy_signal)
        return noise_signal


    def _phase_noise(self, signal):
        '''
        Input:
        signal = 1D PyTorch tensor
        Complex signal
        -------
        Parameters:
        sigma = controls the strength of the phase deviation
        -------
        Returns:
        signal_noise = 1D PyTorch tensor
        Output signal with added phase noise
        '''
        random_sigma = torch.randint(self.sigma_params[TUPLE_INDEX.MIN.value] * 100, self.sigma_params[TUPLE_INDEX.MAX.value]* 100, size=[1])/ 100
        A = random_sigma * torch.randn(signal.shape)
        signal_noise = signal * torch.exp(1j * A)
        return signal_noise


    def _IQ_imbalance (self, signal):
         '''
         Input:
         signal = 1D PyTorch tensor
         Complex signal
         -------
         Parameters:
         gain[db] = amplitude imbalance between I and Q
         phase[deg] = phase mismatch between I and Q
         -------
         Returns: 1D PyTorch tensor
         Output signal with amplitude gain and phase shift to Q.
         '''
         random_gain = torch.randint(self.gain_params[TUPLE_INDEX.MIN.value]*10, self.gain_params[TUPLE_INDEX.MAX.value]*10, size=[1])/10
         random_phase = torch.randint(self.phase_params[TUPLE_INDEX.MIN.value], self.phase_params[TUPLE_INDEX.MAX.value], size=[1])
         gain = 10**(random_gain / 20)
         phase = torch.deg2rad(torch.tensor(random_phase,
                                           dtype=signal.real.dtype,
                                           device=signal.device))
         I = signal.real
         Q = signal.imag
         I_imbalance = I
         Q_imbalance = gain * (torch.cos(phase)*Q + torch.sin(phase)*I)
         return torch.complex(I_imbalance, Q_imbalance)

    def _minor_sat(self, signal):
        '''
        Input:
        signal = 1D PyTorch tensor
        Complex signal
        -------
        Parameter:
        a = scales the input - controls how fast the signal clips
        -------
        Returns: 1D PyTorch tensor
        Output signal at a minor saturation
        '''
        # a = 1.0 / torch.max(torch.abs(signal))
        random_a = torch.randint(self.a_params[TUPLE_INDEX.MIN.value], self.a_params[TUPLE_INDEX.MAX.value], size=[1])/10
        return torch.tanh(signal.real*random_a) + 1j* torch.tanh(signal.imag*random_a)

    def _AM_noise(self, signal):
         '''
         Input:
         signal = 1D PyTorch tensor
         Complex signal
         -------
         Parameters:
         signal: input signal
         cycles: normalized frequency (number of cycles over the full signal)
         min, max = gain fluctuation range (centered around 1)
         p = intensity of amplitude variation (in percent [%])
         -------
         Returns:
         noise * signal = 1D PyTorch tensor
         Output signal with added AM noise
         '''
         n = signal.shape[-1]
         random_cycles = torch.randint(self.cycles_params[TUPLE_INDEX.MIN.value], self.cycles_params[TUPLE_INDEX.MAX.value], size=[1])/100
         random_p = torch.randint(self.p_params[TUPLE_INDEX.MIN.value], self.p_params[TUPLE_INDEX.MAX.value], size=[1])/100
         random_noise = (torch.randint(self.noise_params[TUPLE_INDEX.MIN.value], self.noise_params[TUPLE_INDEX.MAX.value], size=[1])/100
                         * torch.rand(n) + self.noise_params[TUPLE_INDEX.MIN.value])/100
         noise = (1 + random_p * torch.cos(2 * torch.pi * random_cycles * torch.arange(n) / n)
                  * random_noise)
         return noise * signal

    def augment(self, signal):
        '''
        Input:
        signal = 1D PyTorch tensor
        Complex signal
        -------
        Returns:
        noise_signal = 1D PyTorch tensor
        Output signal with added:
            awgn noise
            phase noise
            IQ imbalance
            minor saturation
            AM noise
        '''
        #tensor_signal = torch.from_numpy(signal)
        #noise_signal = self._awgn_noise(signal) ; show_signal(signal, noise_signal)
        # noise_signal = self._phase_noise(signal) ; show_signal(signal, noise_signal)
        # noise_signal = self._IQ_imbalance(signal) ; show_signal(signal, noise_signal)
        # noise_signal = self._minor_sat(signal) ; show_signal(signal, noise_signal)
        noise_signal = self._AM_noise(signal) ; show_signal(signal, noise_signal)
        return noise_signal


if __name__ == "__main__":
    augmentor = Augmentation(
        snr=(5, 13),
        rate= (1,2),
        sigma = (0,1),
        gain_db = (-3, 3),
        phase_deg = (-10, 10),
        a = (0, 20),
        cycles = (0,100),
        p = (0, 20), # p <= 0.2 for realistic noise
        noise = (80, 110) #centered around 1
    )
    num_symbols = 1000

    x_int = np.random.randint(0, 4, num_symbols)  # 0 to 3
    x_degrees = x_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols
    symbols = torch.from_numpy(x_symbols)

    augmented_signal = augmentor.augment(symbols)



