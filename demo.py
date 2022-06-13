import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.masking as mk
import kissdsp.mixing as mx
import kissdsp.reverb as rb
import kissdsp.sink as snk
import kissdsp.source as src
import kissdsp.spatial as sp
import kissdsp.visualize as vz

def demo_waveform():

	# Load speech audio
	ss = src.read("audio/speeches.wav")

	# Display waveforms
	vz.wave(ss)

def demo_spectrogram():

	# Load speech audio
	ss = src.read("audio/speeches.wav")

	# Compute short-time Fourier transform
	Ss = fb.stft(ss)

	# Display spectrograms
	vz.spex(Ss)

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

	# Display room impulse responses
	vz.rir(hs)

def demo_mask():

	# Create a rectangular room with two sources
	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0], [8.0, 7.0, 1.5]]),
	             origin=np.asarray([4.0, 5.0, 1.25]),
	             alphas=0.5 * np.ones(6),
	             c=343.0)

	# Create room impulse responses
	hs = rb.rir(rm)

	# Modify the room to make it anechoic
	rma = rb.anechoic(rm)

	# Create room impulse responses for anechoic room
	hsa = rb.rir(rma)

	# Create room impulse responses for early reflections and late reverb only
	hsr = hs - hsa

	# Load speech audio
	ss = src.read("audio/speeches.wav")

	# Apply room impulse response
	xsa = rb.conv(hsa, ss)
	xsr = rb.conv(hsr, ss)

	# Concatenate
	xsar = mx.concatenate(xsa, xsr)

	# Remix to get anechoic target vs reverb target + interference
	xs = mx.remix(xsar, [[0], [1,2,3]])

	# Get target and residual
	ts = mx.source(xs, 0)
	rs = mx.source(xs, 1)

	# Compute spectrograms
	Ts = fb.stft(ts)
	Rs = fb.stft(rs)

	# Compute masks
	Ms = mk.irm(Ts, Rs)

	# Display
	vz.mask(Ms)


def demo_mvdr():

	# Create a rectangular room with two sources
	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0], [8.0, 7.0, 1.5]]),
	             origin=np.asarray([4.0, 5.0, 1.25]),
	             alphas=0.8 * np.ones(6),
	             c=343.0)

	# Create room impulse responses
	hs = rb.rir(rm)

	# Load speech audio
	ss = src.read("audio/speeches.wav")

	# Generate white noise
	ss[1, :] = 0.01 * src.white(nb_of_samples=ss.shape[1])

	# Apply room impulse response
	xs = rb.conv(hs, ss)

	# Get target and residual
	ts = mx.source(xs, 0)
	rs = mx.source(xs, 1)

	# Get mixture
	ys = mx.collapse(xs)

	# Compute spectrograms
	Ts = fb.stft(ts)
	Rs = fb.stft(rs)
	Ys = fb.stft(ys)

	# Compute spatial correlation matrices
	TTs = sp.scm(Ts)
	RRs = sp.scm(Rs)

	# Compute steering vector
	vs = sp.steering(TTs)

	# Compute mvdr weights
	ws = bf.mvdr(vs, RRs)

	# Perform beamforming
	Zs = bf.beam(Ys, ws)

	# Return to time domain
	zs = fb.istft(Zs)

	# Save audio
	snk.write(ys, "audio/noisy.wav")
	snk.write(zs, "audio/cleaned.wav")


def main():

	parser = ap.ArgumentParser(description='Choose demo.')
	parser.add_argument('--operation', choices=['waveform', 'spectrogram', 'reverb', 'mask', 'mvdr'])
	args = parser.parse_args()

	if args.operation == 'waveform':
		demo_waveform()

	if args.operation == 'spectrogram':
		demo_spectrogram()

	if args.operation == 'reverb':
		demo_reverb()

	if args.operation == 'mask':
		demo_mask()

	if args.operation == 'mvdr':
		demo_mvdr()

if __name__ == "__main__":
	main()




