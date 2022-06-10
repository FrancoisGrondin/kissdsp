import kissdsp.source as src
import kissdsp.visualize as vz

def demo_waveform():

	ss = src.read("audio/speeches.wav")
	vz.wave(ss)

def demo_spectrogram():

	ss = src.read("audio/speeches.wav")

def main():

	demo_waveform()

if __name__ == "__main__":
	main()




