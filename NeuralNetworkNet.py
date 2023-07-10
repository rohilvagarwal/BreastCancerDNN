import torch.nn as nn

class NeuralNetworkNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden1 = nn.Linear(29, 64)
		self.act1 = nn.ReLU()
		self.hidden2 = nn.Linear(64, 128)
		self.act2 = nn.ReLU()
		self.hidden3 = nn.Linear(128, 128)
		self.act3 = nn.ReLU()
		self.hidden4 = nn.Linear(128, 32)
		self.act4 = nn.ReLU()
		self.output = nn.Linear(32, 1)
		self.act_output = nn.Sigmoid()

	def forward(self, x):
		x = self.act1(self.hidden1(x))
		x = self.act2(self.hidden2(x))
		x = self.act3(self.hidden3(x))
		x = self.act4(self.hidden4(x))
		x = self.act_output(self.output(x))
		return x