import numpy as np
import matplotlib.pyplot as plt

def gccphat(XXs, eps=1e-20):
	"""
	Perform Generalized Cross-Correlation with Phase Transform

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins).
        eps (float):
        	Small positive value to avoid division by 0.

    Returns:
    	(np.ndarray):
    		The cross correlation (nb_of_channels, nb_of_channels, nb_of_frames, frame_size).
	"""

	nb_of_channels = XXs.shape[0]
	nb_of_frames = XXs.shape[2]
	nb_of_bins = XXs.shape[3]
	frame_size = (nb_of_bins-1)*2

	xxs = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames, frame_size), dtype=np.float32)

	for channel_index1 in range(0, nb_of_channels):

		for channel_index2 in range(0, nb_of_channels):

			XX = XXs[channel_index1, channel_index2, :, :]
			xxs[channel_index1, channel_index2, :, :] = np.fft.irfft(XX / (np.abs(XX) + eps))

	return xxs


def srpphat(XXs, tdoas, eps=1e-20):
	"""
	Perform Steered-Response Power Phase transform beamforming

	Args:
		XXs (np.ndarray):
			The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins).
		tdoas (np.ndarray):
			Time difference of arrivals (nb_of_doas, nb_of_channels).
		eps (float):
			Small positive value to avoid division by 0.

	Returns:
		(np.ndarray):
			The power in each direction (nb_of_frames, nb_of_doas)
	"""

	nb_of_channels = XXs.shape[0]
	nb_of_frames = XXs.shape[2]
	nb_of_bins = XXs.shape[3]
	nb_of_doas = tdoas.shape[0]
	frame_size = (nb_of_bins-1)*2

	ks = np.expand_dims(np.arange(0, nb_of_bins), axis=1)
	Es = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)

	for channel_index1 in range(0, nb_of_channels):

		taus1 = np.expand_dims(tdoas[:, channel_index1], axis=0)
		Ws1 = np.exp(1j * 2.0 * np.pi * ks @ taus1 / frame_size)

		for channel_index2 in range(0, nb_of_channels):

			taus2 = np.expand_dims(tdoas[:, channel_index2], axis=0)
			Ws2 = np.exp(1j * 2.0 * np.pi * ks @ taus2 / frame_size)

			Ws = Ws1 * np.conj(Ws2)
			XX = XXs[channel_index1, channel_index2, :, :]
			Es += np.clip(np.real((XX / (np.abs(XX) + eps)) @ Ws), a_min=0.0, a_max=None)

	Es /= frame_size * nb_of_channels * (nb_of_channels-1) / 2

	return Es


def srpfast(xxs, tdoas, kernel_lobe=0):
	"""
	Perform Steered-Response Power Phase transform beamforming using cross-correlation

	Args:
		xxs (np.ndarray):
			The cross correlation (nb_of_channels, nb_of_channels, nb_of_frames, frame_size).
		tdoas (np.ndarray):
			Time difference of arrivals (nb_of_doas, nb_of_channels).
		kernel_lobe (int):
			Number of samples on each size of tdoa to get maximum value.

	Returns:
		(np.ndarray):
			The power in each direction (nb_of_frames, nb_of_doas)
	"""

	nb_of_channels = xxs.shape[0]
	nb_of_frames = xxs.shape[2]
	frame_size = xxs.shape[3]
	nb_of_doas = tdoas.shape[0]

	Es = np.zeros((nb_of_frames, nb_of_doas), dtype=np.float32)

	for channel_index1 in range(0, nb_of_channels):

		taus1 = tdoas[:, channel_index1]

		for channel_index2 in range((channel_index1+1), nb_of_channels):

			taus2 = tdoas[:, channel_index2]

			xx = xxs[channel_index1, channel_index2, :, :]

			for kernel in range(0, kernel_lobe):

				xx = np.maximum(xx, np.roll(xx, +1*(kernel+1), axis=1))
				xx = np.maximum(xx, np.roll(xx, -1*(kernel+1), axis=1))

			taus12 = (taus1 - taus2).astype(dtype=int)

			Es += np.take(xx, taus12, axis=1)

	Es /= nb_of_channels * (nb_of_channels-1) / 2

	return Es


def music(XXs, tdoas):
	"""
	Perform Multiple Signal Classification (MUSIC)

	Args:
		XXs (np.ndarray):
			The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins).
		tdoas (np.ndarray):
			Time difference of arrivals (nb_of_doas, nb_of_channels).
	"""

	pass


