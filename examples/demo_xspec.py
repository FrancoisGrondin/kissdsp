import argparse as ap

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave', type=str, default='')
parser.add_argument('--ch1', type=int, default=-1)
parser.add_argument('--ch2', type=int, default=-1)
args = parser.parse_args()

# Load speech audio
ss = io.read(args.wave)

# Compute short-time Fourier transform
Ss = fb.stft(ss)

# Compute cross-spectrum
SSs = sp.xspec(Ss)

# Select the channels to display
if args.ch1 != -1:
	SSs = SSs[[args.ch1], :, :, :]
if args.ch2 != -1:
	SSs = SSs[:, [args.ch2], :, :]

# Display spectrograms
vz.phase(SSs)