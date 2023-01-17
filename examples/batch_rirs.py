import argparse as ap
import json
import numpy as np

import kissdsp.io as io
import kissdsp.micarray as ma
import kissdsp.mixing as mx
import kissdsp.reverb as rb
import kissdsp.visualize as vz

from tqdm import tqdm

# Parse arguments
parser = ap.ArgumentParser(description='Generate RIRs in batch.')
parser.add_argument('--root', type=str, default='', help='Root directory to save RIRs')
parser.add_argument('--micarray', type=str, choices=['respeaker_usb', 'respeaker_core', 'matrix_creator', 'matrix_voice', 'minidsp_uma'], default='', help='Microphone array geometry')
parser.add_argument('--roomsize', type=str, choices=['small', 'medium', 'large'], default='', help='Room size')
parser.add_argument('--alpha', type=float, default=1.0, help='Absorption coefficient')
parser.add_argument('--count', type=int, default=1, help='Number or RIRs to simulate')
args = parser.parse_args()

# Load microphone array
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

# Choose room size
if args.roomsize == 'small':
    box = np.asarray([5.0, 5.0, 2.5])
if args.roomsize == 'medium':
    box = np.asarray([10.0, 10.0, 2.5])
if args.roomsize == 'large':
    box = np.asarray([20.0, 20.0, 4.0])

# Set alpha parameter
alphas = args.alpha * np.ones(6)

# Set speed of sound
c = 343.0

# Margin between receivers/sources and surfaces
margin = 0.1

for index in tqdm(range(1, args.count+1)):

    # Position target source
    src_target_x = np.random.uniform(low=margin, high=box[0]-margin)
    src_target_y = np.random.uniform(low=margin, high=box[1]-margin)
    src_target_z = np.random.uniform(low=margin, high=box[2]-margin)
    
    # Position interfering source
    src_interf_x = np.random.uniform(low=margin, high=box[0]-margin)
    src_interf_y = np.random.uniform(low=margin, high=box[1]-margin)
    src_interf_z = np.random.uniform(low=margin, high=box[2]-margin)    
    
    # Position receiver
    rcv_origin_x = np.random.uniform(low=margin, high=box[0]-margin)
    rcv_origin_y = np.random.uniform(low=margin, high=box[1]-margin)
    rcv_origin_z = np.random.uniform(low=margin, high=box[2]-margin)

    # Create sources and receiver
    srcs = np.asarray([[src_target_x, src_target_y, src_target_z], [src_interf_x, src_interf_y, src_interf_z]])
    origin = np.asarray([rcv_origin_x, rcv_origin_y, rcv_origin_z])

    # Create a rectangular room with one source
    rm = rb.room(mics=mics, box=box, srcs=srcs, origin=origin, alphas=alphas, c=c)

    # Generate room impulse responses
    hs = rb.rir(rm)
    hs_target = hs[0,:,:]
    hs_interf = hs[1,:,:]
    hs = np.concatenate((hs_target, hs_interf), axis=0)

    # Normalize to span full range
    hs = mx.normalize(hs)

    # Write RIRs to file
    io.write(hs, args.root+str(index).zfill(4)+'.wav')

    # Write meta data to file
    with open(args.root+str(index).zfill(4)+'.txt', 'w') as f:
        f.write(str(rm))

