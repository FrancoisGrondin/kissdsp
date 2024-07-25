import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.doas as ds
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.spatial as sp

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave_in', type=str, default='')
parser.add_argument('--wave_out', type=str, default='')
args = parser.parse_args()

# Create mic array
mics_inches = np.asarray([[  -1.00, -15.00,  +0.00 ],
                          [  +7.75,  -6.75,  +0.00 ],
                          [ +14.00, -15.00,  +0.00 ],
                          [  +7.75,  -1.00,  +0.00 ],
                          [ +15.00,  +1.00,  +0.00 ],
                          [  +1.00, +15.00,  +0.00 ],
                          [  +6.75,  +7.75,  +0.00 ],
                          [ +15.00, +14.00,  +0.00 ],
                          [  -1.00,  +7.75,  +0.00 ],
                          [ -14.00, +15.00,  +0.00 ],
                          [  +7.75,  +6.75,  +0.00 ],
                          [  +7.75,  +1.00,  +0.00 ],
                          [ -15.00,  -1.00,  +0.00 ],
                          [  -6.75,  -7.75,  +0.00 ],
                          [ -15.00, -14.00,  +0.00 ],
                          [  +1.00,  -7.75,  +0.00 ]])
mics = mics_inches * 0.0254

# Set direction of interest (unit vector)
doas = np.asarray([[+1.0, +0.0, +0.0]])

# Load recording
xs = io.read(args.wave_in)

# Compute spectrograms
Xs = fb.stft(xs)

# Compute delay and sum weights
ws = sp.steering(sp.freefield(np.squeeze(ds.delay(doas, mics))))

# Perform beamforming
Ys = bf.beam(Xs, ws)

# Return to time domain
ys = fb.istft(Ys)

# Write beamformed signal
io.write(ys, args.wave_out)