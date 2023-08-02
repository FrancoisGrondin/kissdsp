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

def dareit():

	mics = np.zeros((15, 3), dtype=np.float32)

	h = -0.0381

	mics[0, 0] = 0.0635
	mics[0, 1] = 0.0
	mics[0, 2] = 0.0

	mics[1, 0] = 0.1207
	mics[1, 1] = 0.0
	mics[1, 2] = h

	mics[2, 0] = 0.1778
	mics[2, 1] = 0.0
	mics[2, 2] = 2*h

	mics[3, 0] = 0.0196
	mics[3, 1] = -0.0604
	mics[3, 2] = 0.0

	mics[4, 0] = 0.0373
	mics[4, 1] = -0.1147
	mics[4, 2] = h

	mics[5, 0] = 0.0549
	mics[5, 1] = -0.1691
	mics[5, 2] = 2*h

	mics[6, 0] = -0.0514
	mics[6, 1] = -0.0373
	mics[6, 2] = 0.0

	mics[7, 0] = -0.0976
	mics[7, 1] = -0.0709
	mics[7, 2] = h

	mics[8, 0] = -0.1438
	mics[8, 1] = -0.1045
	mics[8, 2] = 2*h

	mics[9, 0] = -0.0514
	mics[9, 1] = 0.0373
	mics[9, 2] = 0.0

	mics[10, 0] = -0.0976
	mics[10, 1] = 0.0709
	mics[10, 2] = h

	mics[11, 0] = -0.1438
	mics[11, 1] = 0.1045
	mics[11, 2] = 2*h

	mics[12, 0] = 0.0196
	mics[12, 1] = 0.0604
	mics[12, 2] = 0.0

	mics[13, 0] = 0.0373
	mics[13, 1] = 0.1147
	mics[13, 2] = h

	mics[14, 0] = 0.0549
	mics[14, 1] = 0.1691
	mics[14, 2] = 2*h

	return mics
	
	