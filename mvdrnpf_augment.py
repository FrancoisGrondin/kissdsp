import argparse as ap
import json as js
import numpy as np
import os as os
import random as rnd

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.localization as loc
import kissdsp.masking as mk
import kissdsp.micarray as ma
import kissdsp.mixing as mix
import kissdsp.reverb as rb
import kissdsp.io as io
import kissdsp.spatial as sp
import kissdsp.transform as tf
import kissdsp.visualize as vz

from tqdm import tqdm

def simulate(file_speech, file_noise, folder, mic_array, nb_of_iterations):

	# Parameters
	box_x_min = 5.0
	box_x_max = 10.0
	box_y_min = 5.0
	box_y_max = 10.0
	box_z_min = 2.0
	box_z_max = 5.0
	alpha_min = 0.2
	alpha_max = 0.8
	c_min = 330.0
	c_max = 355.0
	margin = 0.5
	d_min = 1.0
	d_max = 5.0
	speech_gain_min = 0.0
	speech_gain_max = 20.0
	noise_gain_min = 0.0
	noise_gain_max = 10.0
	vol_min = -20.0
	vol_max = +0.0
	duration = 90000
	pause = 16000

	# Load list of files for speech
	with open(file_speech, 'r') as f:
		files_speech = f.read().splitlines()

	# Load list of files for noise
	with open(file_noise, 'r') as f:
		files_noise = f.read().splitlines()

	for iteration_index in tqdm(range(0, nb_of_iterations)):

		# Choose speech file
		path_speech = os.path.dirname(file_speech) + '/' + rnd.choice(files_speech)

		# Choose noise file
		path_noise = os.path.dirname(file_noise) + '/' + rnd.choice(files_noise)

		# Load speech waveform
		s_speech = tf.window(xs=io.read(path_speech), 
							 duration=duration+pause, 
							 offset=rnd.uniform(0.0, 1.0))

		# Load noise waveform
		s_noise = tf.window(xs=io.read(path_noise),
							duration=duration+pause,
							offset=rnd.uniform(0.0, 1.0))

		# Apply gains to sources
		s_speech = mix.pwr(s_speech, np.random.uniform(speech_gain_min, speech_gain_max, size=(1,)))
		s_noise = mix.pwr(s_noise, np.random.uniform(noise_gain_min, noise_gain_max, size=(1,)))

		# Find a suitable room configuration
		while True:

			# Generate the room size
			box = np.zeros(3, dtype=np.float32)
			box[0] = np.random.uniform(box_x_min, box_x_max)
			box[1] = np.random.uniform(box_y_min, box_y_max)
			box[2] = np.random.uniform(box_z_min, box_z_max)	

			# Generate a random absorption coefficient
			alpha = np.random.uniform(alpha_min, alpha_max) * np.ones(6, dtype=np.float32)

			# Generate a random speed of sound
			c = np.random.uniform(c_min, c_max)

			# Compute random rotation angles
			yaw = np.random.uniform(0, 2*np.pi)
			pitch = np.random.uniform(0, 2*np.pi)
			roll = np.random.uniform(0, 2*np.pi)
			R = rb.rotmat(yaw=yaw, pitch=pitch, roll=roll)

			# Create the microphone array
			if mic_array == 'respeaker_usb':
				mics = ma.respeaker_usb()
			if mic_array == 'respeaker_core':
				mics = ma.respeaker_core()
			if mic_array == 'matrix_creator':
				mics = ma.matrix_creator()
			if mic_array == 'matrix_voice':
				mics = ma.matrix_voice()
			if mic_array == 'minidsp_uma':
				mics = ma.minidsp_uma()
			if mic_array == 'microsoft_kinect':
				mics = ma.microsoft_kinect()

			# Rotate the array
			mics_rot = (R @ mics.T).T

			# Position the array randomly in the room
			origin = np.zeros(3, dtype=np.float32)
			origin[0] = np.random.uniform(0, box[0])
			origin[1] = np.random.uniform(0, box[1])
			origin[2] = np.random.uniform(0, box[2])

			# Position the sound source randomly in the room (target)
			srcs = np.zeros((2, 3), dtype=np.float32)
			srcs[:, 0] = np.random.uniform(0, box[0], size=srcs.shape[0])
			srcs[:, 1] = np.random.uniform(0, box[1], size=srcs.shape[0])
			srcs[:, 2] = np.random.uniform(0, box[2], size=srcs.shape[0])	

			# Create the virtual room
			rm = rb.room(mics=mics_rot, box=box, srcs=srcs, origin=origin, alphas=alpha, c=c)

			# Compute the margin between each source/mics and surfaces
			margin_meas = rb.margin(rm)
			
			# Compute the distances between each source and the mics
			d_min_meas = np.amin(rb.distance(rm))
			d_max_meas = np.amax(rb.distance(rm))

			# If this falls within specifications, then keep room, else loop
			if margin_meas >= margin and d_min_meas >= d_min and d_max_meas <= d_max:
				break

		# Generate room impulse responses
		hs = rb.rir(rm)

		# Split early and late reverberation
		hse, hsl = rb.earlylate(hs)		

		# Get individual RIRs
		speech_hse = np.expand_dims(hse[0, :], 0)
		speech_hsl = np.expand_dims(hsl[0, :], 0)
		noise_hse = np.expand_dims(hse[1, :], 0)
		noise_hsl = np.expand_dims(hsl[1, :], 0)

		# Perform convolutions and crop
		speech_xs_early = tf.crop(rb.conv(speech_hse, s_speech), duration=duration)
		speech_xs_late = tf.crop(rb.conv(speech_hsl, s_speech), duration=duration)
		noise_xs_early = tf.crop(rb.conv(noise_hse, s_noise), duration=duration)
		noise_xs_late = tf.crop(rb.conv(noise_hsl, s_noise), duration=duration)

		# Create target and interference
		target_xs = speech_xs_early
		interf_xs = speech_xs_late + noise_xs_late

		# Apply channelwise gains
		levels = np.random.uniform(-2.0, +2.0, size=mics.shape[0])
		target_xs = mix.gain(target_xs, levels)
		interf_xs = mix.gain(interf_xs, levels)

		# Normalize volume
		xs = target_xs + interf_xs
		maxv = np.amax(np.abs(xs))
		target_xs /= maxv
		interf_xs /= maxv
		vol = rnd.uniform(vol_min, vol_max) * np.ones(mics.shape[0])
		target_xs = mix.gain(target_xs, vol)
		interf_xs = mix.gain(interf_xs, vol)

		# Concatenate
		both_xs = np.concatenate((target_xs, interf_xs), axis=0)

		# Generate unique key
		while True:

			# Generate random key
			key = str(rnd.randint(0, 99999999)).zfill(8)
		
			# Generate directory
			directory = folder + key[0] + '/' + key[1] + '/' + key[2] + '/'

			# Check if it already exists
			path = directory + key + '.wav'
			
			if not os.path.exists(path):
				break

		# Check if directory exists, if not create it
		if not os.path.exists(directory):
			os.makedirs(directory)

		# Save file
		io.write(both_xs, path)

def main():

	parser = ap.ArgumentParser(description='Augment the data.')
	parser.add_argument('--speech', type=str, default='', help='File which holds the list of all speech files.')
	parser.add_argument('--noise', type=str, default='', help='File which holds the list of all noise files.')
	parser.add_argument('--folder', type=str, default='', help='Folder to write the augmented data.')
	parser.add_argument('--micarray', type=str, choices=['respeaker_usb', 'respeaker_core', 'matrix_creator', 'matrix_voice', 'minidsp_uma'])
	parser.add_argument('--nb_of_iterations', type=int, default=0, help='Number of samples generated.')
	args = parser.parse_args()

	simulate(file_speech=args.speech, 
			 file_noise=args.noise,
			 folder=args.folder,
			 mic_array=args.micarray,
			 nb_of_iterations=args.nb_of_iterations)

if __name__ == "__main__":
	main()