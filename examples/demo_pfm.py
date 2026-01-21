import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave', type=str, default='')
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
t = io.read(args.wave)[[0], :]

# Generate white noise
r = 0.01 * np.random.normal(size=t.shape)

# Combine input sources
y = np.concatenate([t,r], axis=0)

# Apply room impulse response
ts = rb.conv(ht, t)
ys = rb.conv(hy, y)

# Compute spectrograms
Ts = fb.stft(ts)
Ys = fb.stft(ys)

# Compute time-difference-of-arrival of target speech
tdoas = rb.tdoa(rm)
tdoa_t = tdoas[0,:]

# Compute oracle spatial correlation matrices
TTs = sp.scm(sp.xspec(Ts))

# Compute steering vectors, one using spatial correlation, another using tdoas
ws_scm = sp.steering(TTs)
ws_tdoa = sp.steering_tdoa(tdoa_t)

# Perform phase-based frequency masking using both steering vectors
Zs_scm = bf.pfm(Ys, ws_scm)
Zs_tdoa = bf.pfm(Ys, ws_tdoa)

# Return to time domain
zs_scm = fb.istft(Zs_scm)
zs_tdoa = fb.istft(Zs_tdoa)

vz.spex(Ys)
vz.spex(Zs_scm)
vz.spex(Zs_tdoa)
vz.wave(ys)
vz.wave(zs_scm)
vz.wave(zs_tdoa)

