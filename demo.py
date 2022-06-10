import argparse as ap
import numpy as np
import kissdsp.filterbank as fb
import kissdsp.mixing as mx
import kissdsp.reverb as rb
import kissdsp.source as src
import kissdsp.visualize as vz

def demo_waveform():

	ss = src.read("audio/speeches.wav")
	vz.wave(ss)

def demo_spectrogram():

	ss = src.read("audio/speeches.wav")
	Ss = fb.stft(ss)
	vz.spex(Ss)

def demo_reverb():

	rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
	             box=np.asarray([10.0, 10.0, 2.5]),
	             srcs=np.asarray([[2.0, 3.0, 1.0]]),
	             origin=np.asarray([4.0, 5.0, 1.25]),
	             alphas=0.5 * np.ones(6),
	             c=343.0)
	hs = rb.rir(rm)

	ss = src.read("audio/speeches.wav")
	xs = mx.collapse(rb.conv(hs, ss))
	vz.wave(xs)


def main():

	parser = ap.ArgumentParser(description='Choose demo.')
	parser.add_argument('--operation', choices=['waveform', 'spectrogram', 'reverb'])
	args = parser.parse_args()

	if args.operation == 'waveform':
		demo_waveform()

	if args.operation == 'spectrogram':
		demo_spectrogram()

	if args.operation == 'reverb':
		demo_reverb()

if __name__ == "__main__":
	main()




