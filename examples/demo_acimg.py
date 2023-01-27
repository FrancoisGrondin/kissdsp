import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.doas as ds
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.localization as lc
import kissdsp.masking as mk
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

import matplotlib.pyplot as plt

# Create mic array
mics = ma.respeaker_usb()

# Parse arguments
parser = ap.ArgumentParser(description='Plot acoustic image.')
parser.add_argument('--wave', type=str, default='')
args = parser.parse_args()

# Create a rectangular room with two sources
rm = rb.room(mics=mics,
             box=np.asarray([10.0, 10.0, 2.5]),
             srcs=np.asarray([[3.0, 3.0, 1.25], [8.0, 7.0, 1.25]]),
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
r = 0.005 * np.random.normal(size=t.shape)

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

# Compute ideal ratio mask
Ms = np.expand_dims(mk.irm(Ts, Rs)[0, :, :], axis=0)

# Compute spatial correlation matrices
YYs_target = sp.oscm(sp.xspec(Ys), Ms)
YYs_interf = sp.oscm(sp.xspec(Ys), 1.0 - Ms)
YYs = sp.oscm(sp.xspec(Ys))
TTs = sp.oscm(sp.xspec(Ts))
RRs = sp.oscm(sp.xspec(Rs))

# Localization
doas = ds.circle()
tdoas = ds.delay(doas, mics)

# Compute projection
SSs = np.linalg.inv(YYs_interf) @ YYs_target

# Reformat shapes
phis_YYs = np.transpose(YYs, (2, 3, 0, 1))
phis_SSs = np.transpose(SSs, (2, 3, 0, 1))

# Perform SRP-PHAT
Es_mix = lc.srpphat(phis_YYs, tdoas)
Es_speech = lc.srpphat(phis_SSs, tdoas)

plt.subplot(2, 1, 1)
plt.imshow(Es_mix, aspect='auto')
plt.plot([53, 53], [1, Es_mix.shape[0]], 'b')
plt.plot([223, 223], [1, Es_mix.shape[0]], 'r')
plt.subplot(2, 1, 2)
plt.imshow(Es_speech, aspect='auto')
plt.plot([53, 53], [1, Es_mix.shape[0]], 'b')
plt.plot([223, 223], [1, Es_mix.shape[0]], 'r')
plt.show()






