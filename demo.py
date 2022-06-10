import argparse
import kissdsp.filterbank as fb
import kissdsp.source as src
import kissdsp.visualize as vz

def demo_waveform():

	ss = src.read("audio/speeches.wav")
	vz.wave(ss)

def demo_spectrogram():

	ss = src.read("audio/speeches.wav")
	Ss = fb.stft(ss)
	vz.spex(Ss)

def main():

	parser = argparse.ArgumentParser(description='Choose demo.')
	parser.add_argument('--operation', choices=['waveform', 'spectrogram'])
	args = parser.parse_args()

	if args.operation == 'waveform':
		demo_waveform()

	if args.operation == 'spectrogram':
		demo_spectrogram()

if __name__ == "__main__":
	main()




