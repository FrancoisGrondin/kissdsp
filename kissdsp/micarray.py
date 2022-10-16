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
	mics[0, 1] = +0.000
	mics[0, 2] = +0.000

	mics[1, 0] = +0.000
	mics[1, 1] = -0.032
	mics[1, 2] = +0.000

	mics[2, 0] = +0.032
	mics[2, 1] = +0.000
	mics[2, 2] = +0.000

	mics[3, 0] = +0.000
	mics[3, 1] = +0.032
	mics[3, 2] = +0.000

	return mics


def respeaker_core():

	mics = np.zeros((6, 3), dtype=np.float32)

	mics[0, 0] = -0.023
	mics[0, 1] = +0.040
	mics[0, 2] = +0.000

	mics[1, 0] = -0.046
	mics[1, 1] = +0.000
	mics[1, 2] = +0.000

	mics[2, 0] = -0.023
	mics[2, 1] = -0.040
	mics[2, 2] = +0.000

	mics[3, 0] = +0.023
	mics[3, 1] = -0.040
	mics[3, 2] = +0.000

	mics[4, 0] = +0.046
	mics[4, 1] = +0.000
	mics[4, 2] = +0.000

	mics[5, 0] = +0.023
	mics[5, 1] = +0.040
	mics[4, 2] = +0.000

	return mics


def matrix_creator():

	mics = np.zeros((8, 3), dtype=np.float32)

	mics[0, 0] = +0.020
	mics[0, 1] = -0.049
	mics[0, 2] = +0.000

	mics[1, 0] = -0.020
	mics[1, 1] = -0.049
	mics[1, 2] = +0.000

	mics[2, 0] = -0.049
	mics[2, 1] = -0.020
	mics[2, 2] = +0.000

	mics[3, 0] = -0.049
	mics[3, 1] = +0.020
	mics[3, 2] = +0.000

	mics[4, 0] = -0.020
	mics[4, 1] = +0.049
	mics[4, 2] = +0.000

	mics[5, 0] = +0.020
	mics[5, 1] = +0.049
	mics[5, 2] = +0.000

	mics[6, 0] = +0.049
	mics[6, 1] = +0.020
	mics[6, 2] = +0.000

	mics[7, 0] = +0.049
	mics[7, 1] = -0.020
	mics[7, 2] = +0.000

	return mics


def matrix_voice():

	mics = np.zeros((8, 3), dtype=np.float32)

	mics[0, 0] = +0.000
	mics[0, 1] = +0.000
	mics[0, 2] = +0.000

	mics[1, 0] = -0.038
	mics[1, 1] = +0.004
	mics[1, 2] = +0.000

	mics[2, 0] = -0.021
	mics[2, 1] = +0.032
	mics[2, 2] = +0.000

	mics[3, 0] = +0.012
	mics[3, 1] = +0.036
	mics[3, 2] = +0.000

	mics[4, 0] = +0.036
	mics[4, 1] = +0.013
	mics[4, 2] = +0.000

	mics[5, 0] = +0.033
	mics[5, 1] = -0.020
	mics[5, 2] = +0.000

	mics[6, 0] = +0.005
	mics[6, 1] = -0.038
	mics[6, 2] = +0.000

	mics[7, 0] = -0.027
	mics[7, 1] = -0.028
	mics[7, 2] = +0.000

	return mics


def minidsp_uma():

	mics = np.zeros((7, 3), dtype=np.float32)

	mics[0, 0] = +0.000
	mics[0, 1] = +0.000
	mics[0, 2] = +0.000

	mics[1, 0] = +0.000
	mics[1, 1] = +0.043
	mics[1, 2] = +0.000

	mics[2, 0] = +0.037
	mics[2, 1] = +0.021
	mics[2, 2] = +0.000

	mics[3, 0] = +0.037
	mics[3, 1] = -0.021
	mics[3, 2] = +0.000

	mics[4, 0] = +0.000
	mics[4, 1] = -0.043
	mics[4, 2] = +0.000

	mics[5, 0] = -0.037
	mics[5, 1] = -0.021
	mics[5, 2] = +0.000

	mics[6, 0] = -0.037
	mics[6, 1] = +0.021
	mics[6, 2] = +0.000

	return mics





