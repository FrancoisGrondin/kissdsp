import numpy as np


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


def introlab_sammy():

	mics = np.zeros((8, 3), dtype=np.float32)

	mics[0, 0] = +0.047
	mics[0, 1] = -0.089
	mics[0, 2] = +0.002

	mics[1, 0] = +0.047
	mics[1, 1] = -0.003
	mics[1, 2] = +0.002

	mics[2, 0] = +0.047
	mics[2, 1] = +0.089
	mics[2, 2] = +0.002

	mics[3, 0] = +0.000
	mics[3, 1] = +0.082
	mics[3, 2] = +0.003

	mics[4, 0] = -0.047
	mics[4, 1] = +0.072
	mics[4, 2] = -0.002

	mics[5, 0] = -0.047
	mics[5, 1] = +0.003
	mics[5, 2] = -0.002

	mics[6, 0] = -0.047
	mics[6, 1] = -0.072
	mics[6, 2] = -0.002

	mics[7, 0] = +0.000
	mics[7, 1] = -0.082
	mics[7, 2] = +0.003

	return mics


	
	