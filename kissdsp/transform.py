import numpy as np


def crop(xs, duration):
	"""
	Crop to the desired number of samples. If too short, pad with zeros.
	
	Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).			

		duration (int):
			Number of samples to keep.

	Returns:
		(np.ndarray):
			Signals in the time domain (nb_of_channels, duration).
	"""

	nb_of_channels = xs.shape[0]
	nb_of_samples = xs.shape[1]

	if nb_of_samples > duration:
		ys = xs[:, 0:duration]
	else:
		ys = np.zeros((nb_of_channels, duration), dtype=np.float32)
		ys[:, 0:nb_of_samples] = xs

	return ys


def roll(xs, offset):
	"""
	Roll the signal to start at the normalized offset.

	Args:
		xs (np.ndarray):
			Signals in the time domain (nb_of_channels, nb_of_samples).

		offset (float):
			Normalized offset (0.0 means beginning of signal, and 1.0 the end).

	Returns:
		(np.ndarray):
			Signals in the time domain (nb_of_channels, nb_of_samples).
	"""

	nb_of_channels = xs.shape[0]
	nb_of_samples = xs.shape[1]

	shift = int(offset * nb_of_samples)

	ys = np.roll(xs, shift)

	return ys


def window(xs, duration, offset):
	"""
	Extract a window from the input signal. 

	If duration is greater than nb_of_samples, then the signal is 
	padded with offset * (duration-nb_of_samples) zeros at the beginning, 
	and (1.0-offset) * (duration-nb_of_samples) at the end.
	
	If duration is less than nb_of_samples, then the signal is
	extracted from sample offset * (nb_of_samples-duration) to sample
	offset * (nb_of_samples-duration) + duration.

	Args:
		xs (np.ndarray):
			Signals in the time domain (nb_of_channels, nb_of_samples).

		duration (int):
			Number of samples in the window.

		offset (float):
			Normalized offset (0.0 means beginning of signal, and 1.0 the end).

	Returns:
		(np.ndarray):
			Signals in the time domain (nb_of_channels, nb_of_samples).
	"""

	nb_of_channels = xs.shape[0]
	nb_of_samples = xs.shape[1]

	if duration > nb_of_samples:

		padding = int(offset * (duration - nb_of_samples))
		ys = np.zeros((nb_of_channels, duration), dtype=np.float32)
		ys[:, padding:(nb_of_samples+padding)] = xs

	else:

		padding = int(offset * (nb_of_samples - duration))
		ys = xs[:, padding:(duration+padding)]

	return ys
