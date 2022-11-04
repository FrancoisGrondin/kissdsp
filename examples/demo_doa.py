import argparse as ap
import numpy as np

import kissdsp.beamformer as bf
import kissdsp.doas as ds
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.localization as lc
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

import matplotlib.pyplot as plt

# Create mic array
mics = ma.respeaker_usb()

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave', type=str, default='')
args = parser.parse_args()

# Create a rectangular room with two sources
rm = rb.room(mics=mics,
             box=np.asarray([10.0, 10.0, 2.5]),
             srcs=np.asarray([[3.0, 3.0, 1.25], [8.0, 7.0, 1.25]]),
             origin=np.asarray([4.0, 5.0, 1.25]),
             alphas=0.5 * np.ones(6),
             c=343.0)

# Create room impulse responses
hy = rb.rir(rm)
ht = hy[[0], :, :]
hr = hy[[1], :, :]

# Load first channel from speech audio
t = io.read(args.wave)[[0], :]

# Generate white noise
r = 0.05 * np.random.normal(size=t.shape)

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

# Compute spatial correlation matrices
TTs = sp.scm(sp.xspec(Ts))
RRs = sp.scm(sp.xspec(Rs))
YYs = sp.scm(sp.xspec(Ys))

# Localization
doas = ds.circle()
tdoas = ds.delay(doas, mics)

SSs = YYs @ np.linalg.inv(RRs)

phis_TTs = np.expand_dims(np.transpose(TTs, (1, 2, 0)), axis=2)
phis_RRs = np.expand_dims(np.transpose(RRs, (1, 2, 0)), axis=2)
phis_YYs = np.expand_dims(np.transpose(YYs, (1, 2, 0)), axis=2)
phis_SSs = np.expand_dims(np.transpose(SSs, (1, 2, 0)), axis=2)

Es_noise = lc.srpphat(phis_RRs, tdoas)
Es_target = lc.srpphat(phis_TTs, tdoas)
Es_mix = lc.srpphat(phis_YYs, tdoas)
Es_filtered = lc.srpphat(phis_SSs, tdoas)

plt.plot(np.squeeze(Es_mix))
plt.axvline(x=53, color='g')
plt.axvline(x=223, color='r')
plt.plot(np.squeeze(Es_filtered))
plt.xlabel('Azimuth')
plt.ylabel('Energy')
plt.show()





