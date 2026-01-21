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

# Get number of channels
n_channels = rm["mics"].shape[0]

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
rs = rb.conv(hr, r)
ys = rb.conv(hy, y)

# Compute spectrograms
Ts = fb.stft(ts)
Rs = fb.stft(rs)
Ys = fb.stft(ys)

# Compute time-difference-of-arrival of target speech
tdoas = rb.tdoa(rm)
tdoa_t = tdoas[0,:]

# Compute oracle spatial correlation matrices
TTs = sp.scm(sp.xspec(Ts))
RRs = sp.scm(sp.xspec(Rs))

# Compute steering vectors, using oracle spatial correlation, as well as tdoas
ws_mvdr_target = bf.mvdr(TTs, RRs)
ws_mvdr_interf = bf.mvdr(RRs, TTs)
ws_tdoa = sp.steering_tdoa(tdoa_t)

# Perform MVDR as reference
Zs_mvdr_target = bf.beam(Ys, ws_mvdr_target)
Zs_mvdr_interf = bf.beam(Ys, ws_mvdr_interf)

# Perform phase-based frequency masking as reference
Zs_tdoa = bf.pfm(Ys, ws_tdoa)

# use phase-based frequency masking to create a mask
Zs_mask = bf.pfm(Ys, ws_tdoa, return_mask=True)

# create masked versions of the input using phase-based frequency masks
target_mask = np.tile(Zs_mask[0,:,:],(n_channels,1,1))
interf_mask = np.tile(Zs_mask[1,:,:],(n_channels,1,1))
Ys_masked_target = Ys * target_mask
Ys_masked_interf = Ys * interf_mask

# use masked input to create spatial correlation matrices
TTs_mask = sp.diagload(sp.scm(sp.xspec(Ys_masked_target)))
RRs_mask = sp.diagload(sp.scm(sp.xspec(Ys_masked_interf)))

# Perform MVDR with "masked" correlation matrices
ws_mvdr_mask_target = bf.mvdr(TTs_mask, RRs_mask)
ws_mvdr_mask_interf = bf.mvdr(RRs_mask, TTs_mask)
Zs_mask_target = bf.beam(Ys, ws_mvdr_mask_target)
Zs_mask_interf = bf.beam(Ys, ws_mvdr_mask_interf)

# Return to time domain
zs_mvdr_target = fb.istft(Zs_mvdr_target)
zs_mvdr_interf = fb.istft(Zs_mvdr_interf)
zs_mask_target = fb.istft(Zs_mask_target)
zs_mask_interf = fb.istft(Zs_mask_interf)
zs_tdoa = fb.istft(Zs_tdoa)

#vz.spex(Ys)
#vz.spex(np.concatenate((Zs_mvdr_target,Zs_mask_target),axis=0))
#vz.spex(np.concatenate((Zs_mvdr_interf,Zs_mask_interf),axis=0))
#vz.wave(ys)
#vz.wave(np.concatenate((zs_mvdr_target,zs_mask_target),axis=0))
#vz.wave(np.concatenate((zs_mvdr_interf,zs_mask_interf),axis=0))

io.write(ys[[0],:]*20,"mic.wav")
io.write(zs_mvdr_target*20,"mvdr_target.wav")
io.write(zs_mvdr_interf*20,"mvdr_interf.wav")
io.write(zs_mask_target*20,"mask_target.wav")
io.write(zs_mask_interf*20,"mask_interf.wav")
io.write(zs_tdoa[[0],:]*20,"pfm_target.wav")
io.write(zs_tdoa[[1],:]*20,"pfm_interf.wav")

