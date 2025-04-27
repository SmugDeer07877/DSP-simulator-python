import torch
import numpy as np
from commpy.channels import awgn
from typing import Tuple
from enum import Enum
from icecream import ic
from Utils import show_signal

class TUPLE_INDEX(Enum):
    MIN = 0
    MAX = 1


class Augmentation:
    """
    A class for applying a variety of signal-level augmentations commonly used in communications and deep learning.

    The augmentations simulate different types of channel and hardware impairments, including:
        - Additive White Gaussian Noise (AWGN)
        - Phase noise
        - IQ imbalance (amplitude and phase mismatch)
        - Minor non-linear saturation
        - Amplitude Modulation (AM) noise

    These augmentations are useful for training robust models that can generalize to real-world noisy signal conditions.

    Parameters
    ----------
    snr : Tuple[int, int]
        Signal-to-noise ratio range in dB for AWGN noise.

    rate : Tuple[float, float]
        Forward error correction (FEC) coding rate range. Used with the AWGN function.

    sigma : Tuple[float, float]
        Range for standard deviation of the phase noise process.

    gain_db : Tuple[float, float]
        Range (in dB) for amplitude mismatch between I and Q in IQ imbalance.

    phase_deg : Tuple[float, float]
        Range (in degrees) for phase mismatch between I and Q in IQ imbalance.

    a : Tuple[float, float]
        Range for non-linearity parameter used in minor saturation.

    cycles : Tuple[int, int]
        Range of cycles for AM noise (affects the frequency of amplitude variation).

    p : Tuple[float, float]
        Percentage intensity of amplitude modulation noise.

    noise : Tuple[int, int]
        Uniform gain variation range (in percent, centered around 100%) for AM noise.

    verbose : int
        Verbosity level for debugging/logging using `icecream`. If > 0, prints intermediate parameters.
    """
    def __init__(self, snr: Tuple = (0,20), sigma: Tuple = (0,0.1),
                 gain_db: Tuple = (-0.5,0.5), phase_deg: Tuple = (-2,2), a: Tuple = (0.1,0.5),
                 cycles:Tuple = (0,20), p:Tuple = (0,0.2), noise:Tuple = (0.9,1.1), verbose: int = 0):
        """
        Initializes the Augmentation object with parameter ranges for various signal impairments.

        Parameters
        ----------
        snr : Tuple[int, int]
            SNR range in dB for AWGN noise (e.g., (5, 13)).

        rate : Tuple[float, float]
            Coding rate range for FEC simulation (usually (1, 1) if no FEC is considered).

        sigma : Tuple[float, float]
            Standard deviation range for phase noise.

        gain_db : Tuple[float, float]
            Amplitude imbalance in dB for IQ imbalance (e.g., (-3, 3)).

        phase_deg : Tuple[float, float]
            Phase mismatch in degrees for IQ imbalance (e.g., (-10, 10)).

        a : Tuple[float, float]
            Saturation parameter range for simulating minor saturation non-linearity.

        cycles : Tuple[int, int]
            Normalized frequency (number of amplitude fluctuation cycles) over the signal length for AM noise.

        p : Tuple[float, float]
            Percentage intensity of amplitude modulation (e.g., (0, 20) for 0–20%).

        noise : Tuple[int, int]
            AM noise gain fluctuation range in percent, centered around 100% (e.g., (80, 110)).

        verbose : int
            If > 0, enables verbose output using `icecream` for debugging.
        """
        self.snr_params = snr
        self.sigma_params = sigma
        self.gain_params = gain_db
        self.phase_params = phase_deg
        self.a_params = a
        self.cycles_params = cycles
        self.p_params = p
        self.noise_params = noise
        self.verbose = verbose
        self.rolls = {}


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
        random_snr = np.random.randint(self.snr_params[TUPLE_INDEX.MIN.value]*100, self.snr_params[TUPLE_INDEX.MAX.value]*100) / 100
        random_rate = 1.0
        if self.verbose:
            ic(random_snr, random_rate)
        noisy_signal = awgn(signal,random_snr, random_rate)
        noise_signal = torch.from_numpy(noisy_signal)
        self.rolls["snr"] = str(random_snr)
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
        random_sigma = torch.randint(int(self.sigma_params[TUPLE_INDEX.MIN.value] * 100), int(self.sigma_params[TUPLE_INDEX.MAX.value]* 100), size=[1])/ 100
        if self.verbose:
            ic(random_sigma)
        A = random_sigma * torch.randn(signal.shape)
        signal_noise = signal * torch.exp(1j * A)
        self.rolls["sigma"] = str(random_sigma.item())
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
         Output signal with amplitude gain and phase shift to I or Q.
         '''
         random_IQ = torch.rand(1)
         random_gain = torch.randint(int(self.gain_params[TUPLE_INDEX.MIN.value]*100), int(self.gain_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
         random_phase = torch.randint(int(self.phase_params[TUPLE_INDEX.MIN.value]*100), int(self.phase_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
         if self.verbose:
             ic(random_gain, random_phase,random_IQ)
         gain = 10**(random_gain / 20)
         phase = torch.deg2rad(torch.tensor(random_phase,
                                           dtype=signal.real.dtype,
                                           device=signal.device))
         I = signal.real
         Q = signal.imag
         if random_IQ:
            I_imbalance = I
            Q_imbalance = gain * (torch.cos(phase)*Q + torch.sin(phase)*I)
         else:
            I_imbalance = gain * (torch.cos(phase)*Q + torch.sin(phase)*I)
            Q_imbalance = Q

         self.rolls["gain"] = str(random_gain.item())
         self.rolls["phase"] = str(random_phase.item())
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
        random_a = torch.randint(int(self.a_params[TUPLE_INDEX.MIN.value]*100), int(self.a_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
        if self.verbose:
            ic(random_a)
        self.rolls["a"] = str(random_a.item())
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
         random_cycles = torch.randint(int(self.cycles_params[TUPLE_INDEX.MIN.value]*100), int(self.cycles_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
         random_p = torch.randint(int(self.p_params[TUPLE_INDEX.MIN.value]*100), int(self.p_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
         noise_value = torch.randint(int(self.noise_params[TUPLE_INDEX.MIN.value]*100), int(self.noise_params[TUPLE_INDEX.MAX.value]*100), size=[1])/100
         random_noise = (noise_value * torch.rand(n) + self.noise_params[TUPLE_INDEX.MIN.value])
         if self.verbose:
             ic(random_cycles, random_p, noise_value)
         noise = (1 + random_p * torch.cos(2 * torch.pi * random_cycles * torch.arange(n) / n)
                  * random_noise)
         self.rolls["cucles"] = str(random_cycles.item())
         self.rolls["p"] = str(random_p.item())
         self.rolls["noise"] = str(noise_value.item())
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
        noise_signal = self._awgn_noise(signal) #; show_signal(signal, noise_signal)
        noise_signal = self._phase_noise(noise_signal) #; show_signal(signal, noise_signal)
        noise_signal = self._IQ_imbalance(noise_signal) #; show_signal(signal, noise_signal)
        noise_signal = self._AM_noise(noise_signal) #; show_signal(signal, noise_signal)
        noise_signal = self._minor_sat(noise_signal) #; show_signal(signal, noise_signal)
        file_name = '_'.join(f'{k}_{v}' for k, v in self.rolls.items())
        print(file_name)
        show_signal(signal, noise_signal, file_name)
        return noise_signal


if __name__ == "__main__":
    augmentor = Augmentation(
        snr=(0, 20),
        sigma = (0,0.1),
        gain_db = (-0.5, 0.5),
        phase_deg = (-2, 2),
        a = (0.1, 0.5),
        cycles = (0,20),
        p = (0, 0.2), # p <= 0.2 for realistic noise
        noise = (0.9, 1.1), #centered around 1
        verbose = 1
    )
    num_symbols = 1000

    x_int = np.random.randint(0, 4, num_symbols)  # 0 to 3
    x_degrees = x_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols
    symbols = torch.from_numpy(x_symbols)

    augmented_signal = augmentor.augment(symbols)



