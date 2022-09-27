import numpy as np
import matplotlib.pyplot as plt


def chirp(f0=100, f1=7000, fS=16000, duration=60.0, nb_of_channels=1):

	c = (f1-f0) / duration
	t = np.arange(0.0, duration, 1.0/fS)

	xs = np.tile(np.sin(2.0 * np.pi * ((c / 2.0) * (t ** 2) + f0 * t)), (nb_of_channels,1))

	return xs


def sweep(Xs, Ys, frame_size=4096):

	nb_of_channels = Ys.shape[0]
	nb_of_frames = Ys.shape[1]
	nb_of_bins = Ys.shape[2]

	Rs = np.zeros((nb_of_channels, nb_of_bins), dtype=np.csingle)
	Cs = np.zeros((nb_of_channels, nb_of_bins), dtype=np.float32)

	for i in range(0, nb_of_channels):
		for j in range(0, nb_of_frames):
			
			Y = Ys[i, j, :]
			X = Xs[i, j, :]
			R = Y / (X + 1e-20)
			
			k = np.argmax(np.abs(Y))

			Rs[i, k] += R[k]
			Cs[i, k] += 1.0

	Hs = Rs / (Cs + 1e-20)
	
	hs = np.fft.irfft(Hs)

	return hs

