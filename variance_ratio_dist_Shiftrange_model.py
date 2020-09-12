import torch
import torch.nn as nn


class Shift_range(nn.Module):
	def __init__(self, low=0, high=1):
		super(Shift_range, self).__init__()
		self.low = low
		self.high = high
		self.range = self.high - self.low

	def forward(self, input):
		mins = torch.min(input, axis = 1, keepdims = True)
		maxs = torch.max(input, axis = 1, keepdims = True)
		out = self.low + ((input - mins.values)*(self.range))/(maxs.values - mins.values)
		
		return out

activation_function = Shift_range()

class BackgroundSmoother(nn.Module):
	def __init__(self):
		super(BackgroundSmoother, self).__init__()

		self.linear = nn.ModuleList(
			[
			 nn.Sequential(nn.Linear(64, 64 * 32), nn.ReLU()),
			 nn.Sequential(nn.Linear(64 * 32, 64 * 32), nn.ReLU()),
			 nn.Sequential(nn.Linear(64 * 32, 64 * 64), Shift_range())
			]
		)
		

	def forward(self, input):
		
		out = input
		for i in range(len(self.linear)):
			out = self.linear[i](out)
		
		out = out.reshape(-1, 1, 64, 64)
		return out
