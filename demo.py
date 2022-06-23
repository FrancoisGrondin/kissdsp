import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.masking as mk
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
	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([6.0, 6.0, 2.5]),
	             srcs=np.asarray([[1.0, 2.0, 1.0]]),
	             origin=np.asarray([3.0, 3.0, 1.25]),
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

	# Display room impulse responses
	vz.rir(hs)
	vz.rir(hse)
	vz.rir(hsl)


def demo_mvdr(file_in, file_out1, file_out2):

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
	r = 0.05 * np.random.normal(size=t.shape)

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

	# Compute steering vector
	vs = sp.steering(TTs)

	# Compute mvdr weights
	ws = bf.mvdr(vs, RRs)

	# Perform beamforming
	Zs = bf.beam(Ys, ws)

	# Return to time domain
	zs = fb.istft(Zs)

	# Save audio
	io.write(ys, file_out1)
	io.write(zs, file_out2)


def main():

	parser = ap.ArgumentParser(description='Choose demo.')
	parser.add_argument('--operation', choices=['waveform', 'spectrogram', 'room', 'reverb', 'mask', 'mvdr'])
	parser.add_argument('--in1', type=str, default='')
	parser.add_argument('--out1', type=str, default='')
	parser.add_argument('--out2', type=str, default='')
	args = parser.parse_args()

	if args.operation == 'waveform':
		demo_waveform(args.in1)

	if args.operation == 'spectrogram':
		demo_spectrogram(args.in1)

	if args.operation == 'room':
		demo_room()

	if args.operation == 'reverb':
		demo_reverb()

	if args.operation == 'mvdr':
		demo_mvdr(args.in1, args.out1, args.out2)

if __name__ == "__main__":
	main()




