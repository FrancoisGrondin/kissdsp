import argparse as ap
import json as js
import numpy as np
import os as os
import random as rnd
import matplotlib.pyplot as plt

import kissdsp.beamformer as bf
import kissdsp.calibration as cb
import kissdsp.filterbank as fb
import kissdsp.localization as loc
import kissdsp.masking as mk
import kissdsp.micarray as ma
import kissdsp.mixing as mix
import kissdsp.reverb as rb
import kissdsp.io as io
import kissdsp.spatial as sp
import kissdsp.visualize as vz

def demo_waveform(file_in):

	# Load speech audio
	ss = io.read(file_in)

	# Display waveforms
	vz.wave(ss)

def demo_spectrogram(file_in):

	# Load speech audio
	ss = io.read(file_in)

	# Compute short-time Fourier transform
	Ss = fb.stft(ss)

	# Display spectrograms
	vz.spex(Ss)

def demo_room():

	# Create a rectangular room with one source
	rm = rb.room(mics=ma.minidsp_uma(),
	             box=np.asarray([3.0, 3.0, 2.5]),
	             srcs=np.asarray([[1.0, 2.0, 1.0], [2.0, 1.0, 1.5]]),
	             origin=np.asarray([1.5, 1.5, 1.25]),
	             alphas=0.5 * np.ones(6),
	             c=343.0)

	# Display room configuration
	vz.room(rm)

def demo_reverb():

	# Create a rectangular room with one source
	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0]]),
	             origin=np.asarray([4.0, 5.0, 1.25]),
	             alphas=0.5 * np.ones(6),
	             c=343.0)
	
	# Generate room impulse responses
	hs = rb.rir(rm)

	# Split early and late reverberation
	hse, hsl = rb.earlylate(hs)

	#np.save('rir.npy', hs[0, 0, :])

	# Display room impulse responses
	vz.rir(hs)
	vz.rir(hse)
	vz.rir(hsl)

def demo_mvdr(file_in):

	# Create a rectangular room with two sources
	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0], [8.0, 7.0, 1.5]]),
	             origin=np.asarray([4.0, 5.0, 1.25]),
	             alphas=0.8 * np.ones(6),
	             c=343.0)

	# Create room impulse responses
	hy = rb.rir(rm)
	ht = hy[[0], :, :]
	hr = hy[[1], :, :]

	# Load first channel from speech audio
	t = io.read(file_in)[[0], :]

	# Generate white noise
	r = 0.01 * np.random.normal(size=t.shape)

	# Combine input sources
	y = np.concatenate([t,r], axis=0)

	# Apply room impulse response
	ts = rb.conv(ht, t)
	rs = rb.conv(hr, r)
	ys = rb.conv(hy, y)

	# Compute spectrograms
	Ts = fb.stft(ts)
	Rs = fb.stft(rs)
	Ys = fb.stft(ys)

	# Compute spatial correlation matrices
	TTs = sp.scm(sp.xspec(Ts))
	RRs = sp.scm(sp.xspec(Rs))

	# Compute mvdr weights
	ws = bf.mvdr(TTs, RRs)

	# Perform beamforming
	Zs = bf.beam(Ys, ws)

	vz.spex(Ys)
	vz.spex(Zs)

	# Return to time domain
	zs = fb.istft(Zs)


def demo_gccphat(file_in):

	# Load input audio
	xs = io.read(file_in)

	# Compute spectrograms
	Xs = fb.stft(xs)

	# Compute cross-spectrum
	XXs = sp.xspec(Xs)

	# Compute cross-correlation
	xxs = loc.gccphat(XXs)

	# Display cross-correlation
	vz.xcorr(xxs)


def demo_mixing(file_in):

	# Load input audio
	xs = io.read(file_in)

	# Power levels
	pwr_levels = np.asarray([0.0, 0.0, 0.0, 0.0])

	# Gain levels
	gain_levels = np.asarray([-3.0, -3.0, -3.0, -3.0])

	# Mix
	ys = mix.gain(mix.normalize(mix.pwr(xs, pwr_levels)), gain_levels)

	vz.wave(xs)
	vz.wave(ys)


def demo_calibration():


	# Create a rectangular room with one source
	rm = rb.room(mics=np.asarray([[0.0, 0.0, 0.0]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0]]),
	             origin=np.asarray([2.3, 3.2, 1.1]),
	             alphas=0.5 * np.ones(6),
	             c=343.0)
	
	# Generate room impulse responses
	hs = rb.rir(rm)

	# Split early and late reverberation
	hse, hsl = rb.earlylate(hs)

	# Create excitation signal
	xs = cb.chirp(duration=300.0)	
	
	# Convolve 
	ys = rb.conv(hs, xs)
	ys += np.random.normal(scale=0.01, size=ys.shape)
	
	# STFTs
	Xs = fb.stft(xs, frame_size=4096, hop_size=512)
	Ys = fb.stft(ys, frame_size=4096, hop_size=512)

	hsEst = cb.sweep(Xs, Ys)

	plt.subplot(2, 1, 1)
	plt.plot(hs[0, 0, :])
	plt.subplot(2, 1, 2)
	plt.plot(hsEst[0, :])
	plt.show()

def main():

	parser = ap.ArgumentParser(description='Choose demo.')
	parser.add_argument('--operation', choices=['waveform', 'spectrogram', 'room', 'reverb', 'mask', 'mvdr', 'gccphat', 'mix', 'calibration'])
	parser.add_argument('--wave', type=str, default='')
	args = parser.parse_args()

	if args.operation == 'waveform':
		demo_waveform(file_in=args.wave)

	if args.operation == 'spectrogram':
		demo_spectrogram(file_in=args.wave)

	if args.operation == 'room':
		demo_room()

	if args.operation == 'reverb':
		demo_reverb()

	if args.operation == 'mvdr':
		demo_mvdr(file_in=args.wave)

	if args.operation == 'gccphat':
		demo_gccphat(file_in=args.wave)

	#if args.operation == 'mix':
	#	demo_mixing(file_in=args.in1)

	if args.operation == 'calibration':
		demo_calibration()

if __name__ == "__main__":
	main()




