import argparse as ap
import matplotlib.pyplot as plt
import numpy as np

import kissdsp.doas as ds
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.localization as lc
import kissdsp.micarray as ma
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--wave', type=str, default='')
args = parser.parse_args()

# Create mic array
mics = np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]])

# Create sphere around the array
doas = ds.sphere()

# Generate TDoAs for freefield propagation
tdoas = ds.delay(doas=doas, mics=mics)

# Load the signals
xs = io.read(args.wave)

# Compute short-time Fourier transform
Xs = fb.stft(xs)

# Compute the cross-spectrum
XXs = sp.xspec(Xs)

# Compute the cross-correlation
xxs = lc.gccphat(XXs)

# Compute the acoustic images
acimg = lc.srpfast(xxs, tdoas, kernel_lobe=1)

# Get potential sources
pots = doas[np.argmax(acimg, axis=1), :]

# Get energy
Es = np.amax(acimg, axis=1)

# Get elevation and azimuth in degrees
azs = np.arctan2(pots[:, 1], pots[:, 0]) / np.pi * 180.0
els = np.arcsin(pots[:, 2]) / np.pi * 180.0

# Get frame indexes
ts = np.arange(0, azs.shape[0])

ax = plt.subplot(3, 1, 1)
ax.set_title("Azimuth")
ax.set_ylim([-180.0, +180.0])
ax.scatter(ts, azs)
ax = plt.subplot(3, 1, 2)
ax.set_title("Elevation")
ax.set_ylim([-90.0, +90.0])
ax.scatter(ts, els)
ax = plt.subplot(3, 1, 3)
ax.set_title("Energy")
ax.set_ylim([+0.0, +1.0])
ax.plot(ts, Es)
plt.show()


