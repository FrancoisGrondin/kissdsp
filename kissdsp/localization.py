import numpy as np

def gccphat(XXs, eps=1e-20):
	"""
	Perform Generalized Cross-Correlation with Phase Transform

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins).
        eps (float):
        	Small positive value to avoid division by 0.

    Returns:
    	xxs (np.ndarray):
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