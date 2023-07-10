import torch
from torch.utils.data import Dataset

class DNNDataLoader(Dataset):
	def __init__(self, df):
		self.df = df

		self.dfInput = df[:, :-1]
		self.dfOutput = df[:, -1]

		self.dfInputTensor = torch.tensor(self.dfInput, dtype=torch.float)
		self.dfOutputTensor = torch.tensor(self.dfOutput, dtype=torch.float)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx): #used in indexing
		return self.dfInputTensor[idx], self.dfOutputTensor[idx]