import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.masking as mk
import kissdsp.spatial as sp
import kissdsp.visualize as vz

import argparse as ap
import numpy as np
import os as os

from tqdm import tqdm

class Audio(data.Dataset):

	def __init__(self, file_meta, frame_size, hop_size):

		self.root = os.path.dirname(file_meta)
		self.frame_size = frame_size
		self.hop_size = hop_size

		with open(file_meta) as f:
			self.elements = f.read().splitlines()

	def __len__(self):

		return len(self.elements)

	def __getitem__(self, idx):

		xs = io.read(self.root + "/" + self.elements[idx])

		nb_of_channels_times_2 = xs.shape[0]
		nb_of_channels = int(nb_of_channels_times_2 / 2)

		# Extract target and interference
		xs_target = xs[:nb_of_channels, :]
		xs_interf = xs[nb_of_channels:, :]
		xs_all = xs_target + xs_interf

		# Compute STFTs
		Xs_target = fb.stft(xs_target, hop_size=self.hop_size, frame_size=self.frame_size)
		Xs_interf = fb.stft(xs_interf, hop_size=self.hop_size, frame_size=self.frame_size)
		Xs_all = fb.stft(xs_all, hop_size=self.hop_size, frame_size=self.frame_size)

		# Compute SCMs
		XXs_target = sp.scm(sp.xspec(Xs_target))
		XXs_interf = sp.scm(sp.xspec(Xs_interf))		

		# Compute steering vector
		vs = sp.steering(XXs_target)

		# Compute mvdr weights
		ws = bf.mvdr(vs, XXs_interf)

		# Perform beamforming
		Ys_target = bf.beam(Xs_target, ws)
		Ys_interf = bf.beam(Xs_interf, ws)
		Ys_all = bf.beam(Xs_all, ws)
		Ps_all = bf.avgpwr(Xs_all, ws)

		# Compute ideal ratio mask
		Ms_all = mk.irm(Ys_target, Ys_interf)

		# Generate features
		clean = np.squeeze(Ys_target, axis=0)
		beam = np.squeeze(Ys_all, axis=0)
		avg = np.squeeze(Ps_all, axis=0)
		mask = np.squeeze(Ms_all, axis=0)

		return clean, beam, avg, mask


class Network(nn.Module):

	def __init__(self, frame_size, hidden_size, num_layers, dropout):

		super(Network, self).__init__()

		self.frame_size = frame_size
		self.hidden_size = hidden_size

		self.bn = nn.BatchNorm2d(num_features=2)

		self.lstm = nn.LSTM(input_size=int(self.frame_size/2+1)*2, 
							hidden_size=self.hidden_size, 
							num_layers=num_layers,
							dropout=dropout,
							batch_first=True,
							bidirectional=True)

		self.fc = nn.Conv2d(in_channels=self.hidden_size*2,
							out_channels=int(self.frame_size/2+1),
							kernel_size=1)

	def forward(self, beams, avgs):

		# Unsqueeze: N x T x F > N x 1 x T x F
		beams = torch.unsqueeze(beams, dim=1)

		# Unsqueeze: N x T x F > N x 1 x T x F
		avgs = torch.unsqueeze(avgs, dim=1)
		
		# Concatenate (N x 1 x T x F) & (N x 1 x T x F) > N x 2 x T x F
		x = torch.cat((beams, avgs) , dim=1)

		# Compute amplitude in dB
		x = torch.log(torch.abs(x) ** 2 + 1e-10) - torch.log(torch.abs(x) * 0 + 1e-10)

		# Batch norm: N x 2 x T x F > N x 2 x T x F
		x = self.bn(x)

		# Permute: N x 2 x T x F > N x T x F x 2
		x = x.permute(0,2,3,1)

		# View: N x T x F x 2 > N x T x 2F
		x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

		# LSTM: N x T x 2F > N x T x 2H
		x, _ = self.lstm(x)

		# Permute: N x T x 2H > N x 2H x T
		x = x.permute(0,2,1)

		# Unsqueeze: N x 2H x T > N x 2H x T x 1
		x = torch.unsqueeze(x, 3)

		# FC: N x 2H x T x 1 > N x F x T x 1
		x = self.fc(x)

		# Permute: N x F x T x 1 > N x 1 x T x F
		x = x.permute(0,3,2,1)

		# Squeeze: N x 1 x T x F > N x T x F
		x = torch.squeeze(x, dim=1)

		# Set between 0 and 1
		x = torch.sigmoid(x)

		return x

class Loss:

	def __init__(self):

		self.mseloss = nn.MSELoss()

	def __call__(self, beams, masks_pred, masks_target):

		beams = torch.log(torch.abs(beams) ** 2 + 1e-10) - torch.log(torch.abs(beams) * 0 + 1e-10)

		return self.mseloss(masks_pred * beams, masks_target * beams)

class Brain:

	def __init__(self, dataset_train, dataset_eval, dataset_test, num_workers, shuffle, batch_size, frame_size, hop_size, hidden_size, num_layers, dropout, diffcoh):

		# Save hyperparameters
		self.batch_size = batch_size
		self.frame_size = frame_size
		self.hop_size = hop_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.diffcoh = diffcoh

		# Get CUDA if possible
		torch.backends.cudnn.enabled = False
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		# Create datasets
		self.dset_train = Audio(file_meta=dataset_train, frame_size=frame_size, hop_size=hop_size)
		self.dset_eval = Audio(file_meta=dataset_eval, frame_size=frame_size, hop_size=hop_size)
		self.dset_test = Audio(file_meta=dataset_test, frame_size=frame_size, hop_size=hop_size)

		# Create dataloaders
		self.dload_train = data.DataLoader(self.dset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
		self.dload_eval = data.DataLoader(self.dset_eval, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
		self.dload_test = data.DataLoader(self.dset_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

		# Create model, loss and optimizer
		self.net = Network(frame_size=frame_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(self.device)
		self.criterion = Loss()
		self.optimizer = optim.Adam(self.net.parameters())

	def load(self, file_network):

		pass

	def save (self, file_network):

		pass

	def train(self):

		# Total loss
		total_loss = 0.0

		# Enable back prop
		self.net.train()

		for _, beams, avgs, masks in tqdm(self.dload_train):

			# Transfer to device (CPU or GPU)
			beams = beams.to(self.device)
			avgs = avgs.to(self.device)
			masks_target = masks.to(self.device)

			# Zero gradients
			self.optimizer.zero_grad()

			# Forward
			if self.diffcoh == True:
				masks_pred = self.net(beams, avgs)
			else:
				masks_pred = self.net(beams, beams)

			# Compute weighted loss
			loss = self.criterion(beams, masks_pred, masks_target)

			# Perform back prop
			loss.backward()

			# Update parameters
			self.optimizer.step()

			# Add to total loss
			total_loss += loss.item()

		# Compute average loss
		avg_loss = total_loss / len(self.dload_train)

		return avg_loss

	def eval(self):

		# Total loss
		total_loss = 0.0

		# Disable back prop
		self.net.eval()

		for _, beams, avgs, masks in tqdm(self.dload_eval):

			# Transfer to device (CPU or GPU)
			beams = beams.to(self.device)
			avgs = avgs.to(self.device)
			masks_target = masks.to(self.device)

			# Forward
			if self.diffcoh == True:
				masks_pred = self.net(beams, avgs)
			else:
				masks_pred = self.net(beams, beams)

			# Compute weighted loss
			loss = self.criterion(beams, masks_pred, masks_target)

			# Add to total loss
			total_loss += loss.item()

		# Compute average loss
		avg_loss = total_loss / len(self.dload_eval)

		return avg_loss

	def test(self):

		# Total PESQ
		total_pesq = 0.0

		# Disable back prop
		self.net.eval()

		for cleans, beams, avgs, masks in tqdm(self.dload_test):

			# Transfer to device (CPU or GPU)
			cleans = cleans.to(self.device)
			beams = beams.to(self.device)
			avgs = avgs.to(self.device)
			masks_target = masks.to(self.device)

			# Estimate mask
			if self.diffcoh == True:
				masks_pred = self.net(beams, avgs)
			else:
				masks_pred = self.net(beams, beams)

			# Get clean signal
			xs_clean = fb.istft(cleans, hop_size=self.hop_size)

			# Get enhanced signal with oracle mask
			xs_oracle = fb.istft(beams * masks_target, hop_size=self.hop_size)

			# Get enhanced signal with predicted mask
			xs_estimated = fb.istft(beams * masks_pred, hop_size=self.hop_size)

	def peek(self, idx, dset):

		# Disable back prop
		self.net.eval()

		# Load data
		if (dset == 'train'):
			clean, beam, avg, mask = self.dset_train[idx]
		if (dset == 'eval'):
			clean, beam, avg, mask = self.dset_eval[idx]
		if (dset == 'test'):
			clean, beam, avg, mask = self.dset_test[idx]

		# Fix dimensions for batch
		cleans = torch.unsqueeze(torch.from_numpy(clean), dim=0)
		beams = torch.unsqueeze(torch.from_numpy(beam), dim=0)
		avgs = torch.unsqueeze(torch.from_numpy(avg), dim=0)
		masks_target = torch.unsqueeze(torch.from_numpy(mask), dim=0)

		# Predict mask
		if self.diffcoh == True:
			masks_pred = self.net(beams, avgs)
		else:
			masks_pred = self.net(beams, beams)

		# Return results
		mask_pred = masks_pred.detach().numpy()
		mask_target = masks_target.detach().numpy()

		return mask_target, mask_pred


def main():

	parser = ap.ArgumentParser(description='Train/use network.')
	parser.add_argument('--dataset_train', type=str, default='')
	parser.add_argument('--dataset_eval', type=str, default='')
	parser.add_argument('--dataset_test', type=str, default='')
	parser.add_argument('--logbook', type=str, default='')
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--frame_size', type=int, default=512)
	parser.add_argument('--hop_size', type=int, default=128)
	parser.add_argument('--hidden_size', type=int, default=128)
	parser.add_argument('--num_layers', type=int, default=2)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--diffcoh', type=bool, default=True)
	args = parser.parse_args()

	brain = Brain(dataset_train=args.dataset_train,
				  dataset_eval=args.dataset_eval,
				  dataset_test=args.dataset_test,
				  num_workers=12,
				  shuffle=True,
				  batch_size=args.batch_size,
				  frame_size=args.frame_size,
				  hop_size=args.hop_size,
				  hidden_size=args.hidden_size,
				  num_layers=args.num_layers,
				  dropout=args.dropout,
				  diffcoh=args.diffcoh)

	for epoch in range(0, args.epochs):

		train_loss = brain.train()
		eval_loss = brain.eval()
	


if __name__ == "__main__":
	main()

