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
parser.add_argument('--micarray', type=str, choices=['respeaker_usb', 'respeaker_core', 'matrix_creator', 'matrix_voice', 'minidsp_uma', 'introlab_sammy'])
args = parser.parse_args()

# Create mic array
if args.micarray == 'respeaker_usb':
	mics = ma.respeaker_usb()
if args.micarray == 'respeaker_core':
	mics = ma.respeaker_core()
if args.micarray == 'matrix_creator':
	mics = ma.matrix_creator()
if args.micarray == 'matrix_voice':
	mics = ma.matrix_voice()
if args.micarray == 'minidsp_uma':
	mics = ma.minidsp_uma()
if args.micarray == 'introlab_sammy':
	mics = ma.introlab_sammy()

# Create sphere around the array
doas = ds.sphere()
doas = doas[np.abs(doas[:, 2]) < 0.5, :]

# Generate TDoAs for freefield propagation
tdoas = ds.delay(doas=doas, mics=mics)

# Load the signals
xs = io.read(args.wave)

# Compute short-time Fourier transform
Xs = fb.stft(xs)

# Compute the cross-correlation
XXs = sp.xspec(Xs)

# Compute the acoustic images
acimg = lc.srpphat(XXs, tdoas)

# Get potential sources
pots = doas[np.argmax(acimg, axis=1), :]

print(tdoas[1053, :])

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


