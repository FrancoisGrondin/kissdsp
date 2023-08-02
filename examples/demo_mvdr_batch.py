import argparse as ap
import numpy as np
import random as rnd
import os
import string as strg

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

# Parse arguments
parser = ap.ArgumentParser(description='Plot wave forms.')
parser.add_argument('--input', type=str, default='', help='Text file that contains the list of speech wave paths.')
parser.add_argument('--output', type=str, default='', help='Output directory to save augmented audio.')
parser.add_argument('--count', type=int, default=1, help='Number of audio files to generate.')
args = parser.parse_args()

# Load list of all speech files
with open(args.input, 'r') as file:
    speech_files = file.readlines()

# Format output directory
output_directory = os.path.join(args.output, '')

# Generate files
for iteration in range(0, args.count):

    # Choose room size
    room_type = rnd.choice(['small','medium','large'])

    if room_type == 'small':

        box = np.asarray([ np.random.uniform(low=1, high=10), 
                           np.random.uniform(low=1, high=10),
                           np.random.uniform(low=2, high=5) ])

    if room_type == 'medium':

        box = np.asarray([ np.random.uniform(low=10, high=30), 
                           np.random.uniform(low=10, high=30),
                           np.random.uniform(low=2, high=5) ])

    if room_type == 'large':

        box = np.asarray([ np.random.uniform(low=30, high=50), 
                           np.random.uniform(low=30, high=50),
                           np.random.uniform(low=2, high=5) ])

    # Choose mic array
    # For now set it to ReSpeaker USB
    mics = ma.respeaker_usb()

    # Position array in room (ensure a margin of 0.5m from walls, floor and ceiling)
    margin = 0.5
    origin = np.asarray([ np.random.uniform(low=margin, high=box[0]-margin),
                          np.random.uniform(low=margin, high=box[1]-margin),
                          np.random.uniform(low=margin, high=box[2]-margin) ])

    # Position sources in room (ensure a margin of 0.5m from wall, floor and ceiling)
    margin = 0.5
    srcs = np.asarray([[ np.random.uniform(low=margin, high=box[0]-margin),
                         np.random.uniform(low=margin, high=box[1]-margin),
                         np.random.uniform(low=margin, high=box[2]-margin) ],
                       [ np.random.uniform(low=margin, high=box[0]-margin),
                         np.random.uniform(low=margin, high=box[1]-margin),
                         np.random.uniform(low=margin, high=box[2]-margin) ]])

    # Select reverberation level
    alphas = np.random.uniform(low=0.2, high=0.8) * np.ones(6)

    # Choose speed of sound (set to a constant for now)
    c = 343.0

    # Create a rectangular room with two sources
    rm = rb.room(mics=mics, box=box, srcs=srcs, origin=origin, alphas=alphas, c=c)

    # Generate room impulse responses
    hs = rb.rir(rm)
    h_target = hs[[0], :, :]
    h_interf = hs[[1], :, :]

    # Get two random paths
    path_target = rnd.choice(speech_files).rstrip('\n')
    path_interf = rnd.choice(speech_files).rstrip('\n')

    # Load speech audio
    s_target = io.read(path_target)
    s_interf = io.read(path_interf)

    # Normalize both to 0 dB
    s_target /= np.mean(s_target ** 2) ** 0.5
    s_interf /= np.mean(s_interf ** 2) ** 0.5

    # Generate SNR
    snr_target = np.random.uniform(-5, +5)
    snr_interf = np.random.uniform(-5, +5)

    # Apply gain for SNR
    snr_target *= 10 ** (snr_target / 20)
    snr_interf *= 10 ** (snr_interf / 20)

    # Apply room impulse response
    x_target = rb.conv(h_target, s_target)
    x_interf = rb.conv(h_interf, s_interf)
    x_target += 0.001 * np.random.normal(size=x_target.shape)
    x_interf += 0.001 * np.random.normal(size=x_interf.shape)
    x = x_target + x_interf

    # Compute spectrograms
    Xs_target = fb.stft(x_target)
    Xs_interf = fb.stft(x_interf)
    Xs = fb.stft(x)

    # Compute spatial correlation matrices
    XXs_target = sp.scm(sp.xspec(Xs_target))
    XXs_interf = sp.scm(sp.xspec(Xs_interf))

    # Compute mvdr weights
    ws_target = bf.mvdr(XXs_target, XXs_interf)
    ws_interf = bf.mvdr(XXs_interf, XXs_target)

    # Perform beamforming
    Ys_target = bf.beam(Xs, ws_target)
    Ys_interf = bf.beam(Xs, ws_interf)
    Ys_ideal = bf.beam(Xs_target, ws_target)

    # Go back to time-domain
    ys_target = fb.istft(Ys_target)
    ys_interf = fb.istft(Ys_interf)
    ys_ideal = fb.istft(Ys_ideal)

    # Span full range
    vol_norm = max([ np.amax(np.abs(ys_target)),
                     np.amax(np.abs(ys_interf)),
                     np.amax(np.abs(ys_ideal)) ])
    ys_target /= vol_norm
    ys_interf /= vol_norm
    ys_ideal /= vol_norm

    # Apply new volume
    volume = 10 ** (np.random.uniform(low=-20, high=0) / 10)
    ys_target *= volume
    ys_interf *= volume
    ys_ideal *= volume

    # Concatenate to a single multi-channel audio signal
    ys = np.concatenate([ ys_target, ys_interf, ys_ideal ], axis=0)

    # Save to file
    io.write(ys, "%s%s.wav" % (output_directory, ''.join(rnd.choices(strg.ascii_lowercase, k=8))))

