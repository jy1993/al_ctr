from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class DeepFM(nn.Module):
	"""docstring for DeepFM"""
	def __init__(self, arg):
		super(DeepFM, self).__init__()
		self.arg = arg
		fm = [nn.Linear(ele, self.args.k) for ele in self.arg.fields]
		num_fields = len(fm)
		self.fm = nn.ModuleList(fm)
		self.fc1 = nn.Linear(num_fields * k, self.args.fc1)
		self.fc2 = nn.Linear(self.args.fc1, 1)

	def forward(self, x):
		cumsum = np.cum_sum(self.arg.fields)
		cut_x = [x[, cumsum[ii]:cumsum[ii + 1]] for ii in len(cumsum)]
		out1 = [F.relu(part(cut_x[ii])) for ii, part in enumerate(self.fm)]
		out2 = out1.concat(axis=1)
		out3 = F.relu(self.fc1(out2))
		out = F.sigmoid(self.fc2(out3))
		return out



		

		