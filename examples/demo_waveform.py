import argparse as ap

import kissdsp.io as io
import kissdsp.visualize as vz

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave', type=str, default='')
args = parser.parse_args()

# Load speech audio
ss = io.read(args.wave)

# Display waveforms
vz.wave(ss)