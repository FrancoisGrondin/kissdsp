import numpy as np


def linear(nb_of_mics, length):
	"""
	Create a mic array with uniformly distributed microphones on a line.

	Args:
		nb_of_mics (int):
			Number of microphones.
		length (float):
			Length of the linear array (in m).
	"""

	xs = np.linspace(-1.0 * length / 2.0, +1.0 * length / 2.0, nb_of_mics)

	mics = np.zeros((nb_of_mics, 3), dtype=np.float32)

	mics[:, 0] = xs

	return mics


def circular(nb_of_mics, diameter):
	"""
	Create a mic array with uniformly distributed microphones on a circle.

	Args:
		nb_of_mics (int):
			Number of microphones.
		diameter (float):
			Diameter of the circular array (in m).

	Return:
		(np.ndarray):
			Microphone positions (nb_of_mics, 3).	
	"""
	
	thetas = 2 * np.pi * np.arange(0, nb_of_mics) / nb_of_mics

	mics = np.zeros((nb_of_mics, 3), dtype=np.float32)

	mics[:, 0] = (diameter/2) * np.cos(thetas)
	mics[:, 1] = (diameter/2) * np.sin(thetas)

	return mics


def respeaker_usb():

	mics = np.zeros((4, 3), dtype=np.float32)

	mics[0, 0] = -0.032
	mics[1, 1] = -0.032
	mics[2, 0] = +0.032
	mics[3, 1] = +0.032

	return mics

