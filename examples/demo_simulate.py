import argparse as ap
import numpy as np

import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.mixing as mx
import kissdsp.reverb as rb

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave_in', type=str, default='')
parser.add_argument('--wave_out', type=str, default='')
args = parser.parse_args()

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
t = io.read(args.wave_in)[[0], :]

# Generate white noise
r = 0.01 * np.random.normal(size=t.shape)

# Combine input sources
y = np.concatenate([t,r], axis=0)

# Apply room impulse response
ys = 0.99 * mx.normalize(rb.conv(hy, y))

# Save
io.write(ys, args.wave_out)