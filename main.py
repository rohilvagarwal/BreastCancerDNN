import numpy as np
from PreProcessData import PreProcessData
import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNetworkNet import NeuralNetworkNet
from DNNDataLoader import DNNDataLoader

#import data
data = np.loadtxt(fname='breast_cancer_data/wdbc.data', dtype=str, delimiter=',')

preprocessedData = PreProcessData(data)

train, test = preprocessedData.get_train_and_test()

trainDataLoader = DNNDataLoader(train)
testDataLoader = DNNDataLoader(test)

model = NeuralNetworkNet()

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

numEpochs = 10
batch_size = 10

for epoch in range(numEpochs):
	print(f"epoch {epoch}")
	for i, (xBatch, yBatch) in enumerate(trainDataLoader):
		yPredToTrain = model(xBatch)

		#print(yPredToTrain, yBatch)

		loss = loss_fn(yPredToTrain.squeeze(), yBatch.squeeze())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

testInput = torch.empty(0, 29)
testOutput = torch.empty(0, 1)

#separate input and output in batches
for batchInput, batchOutput in testDataLoader:
	# print(testInput)
	# print(batchInput)

	print(testOutput)
	print(batchOutput)

	testInput = torch.cat((testInput, batchInput.unsqueeze(0)), dim=0)
	testOutput = torch.cat((testOutput, batchOutput.unsqueeze(0)), dim=0)

#predict values with testing data input
yPred = model(testInput)

#Calculate Mean Absolute Error (MAE)
mae = torch.abs(yPred - testOutput).mean()

#Calculate Coefficient of Determination (r^2)
sumOfSquaresOfResiduals = torch.sum(torch.square(yPred - testOutput))
sumOfSquaresTotal = torch.sum(torch.square(testOutput - torch.mean(testOutput)))
r2 = 1 - (sumOfSquaresOfResiduals / sumOfSquaresTotal)

print(f"MAE {mae}")
print(f"r2 {r2}")