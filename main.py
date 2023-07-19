import numpy as np
from PreProcessData import PreProcessData
import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNetworkNet import NeuralNetworkNet
from DNNDataLoader import DNNDataLoader
import matplotlib.pyplot as plt

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

numEpochs = 30
batch_size = 32

# create testing data
testInput: torch.Tensor = torch.empty(0)
testOutput: torch.Tensor = torch.empty(0)
# testOutput = testOutput.unsqueeze(0)

#separate input and output in batches
for batchInput, batchOutput in testDataLoader:
	#print(batchInput.size())

	testInput = torch.cat((testInput, batchInput), dim=0)
	testOutput = torch.cat((testOutput, batchOutput.unsqueeze(0)), dim=0)

testInput = testInput.reshape(testInput.size()[0] // 29, 29)

# simulate epochs
testAccuracies = []
testLosses = []
for epoch in range(numEpochs):
	print(f"epoch {epoch + 1}")
	for i, (xBatch, yBatch) in enumerate(trainDataLoader):
		yPredToTrain = model(xBatch)

		#print(yPredToTrain, yBatch)

		loss = loss_fn(yPredToTrain.squeeze(), yBatch.squeeze())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Testing
	# predict values with testing data input
	yPred = model(testInput)

	threshold = 0.5  # Adjust the threshold based on your requirements

	# Convert predicted probabilities to binary predictions
	binary_predictions = (yPred >= threshold).int()

	# Convert binary_predictions to float
	binary_predictions = binary_predictions.float().view(-1)
	testOutput = testOutput.view(-1)

	# print(binary_predictions)
	# print(testOutput)

	# Calculate accuracy
	accuracy = (binary_predictions == testOutput).float().mean()
	testAccuracies.append(accuracy.item() * 100)

	# Calculate loss
	loss = loss_fn(yPred.squeeze(), testOutput.squeeze())
	testLosses.append(loss.detach().numpy())

# print(testAccuracies)

# #Calculate Mean Absolute Error (MAE)
# mae = torch.abs(yPred - testOutput).mean()
#
# #Calculate Coefficient of Determination (r^2)
# sumOfSquaresOfResiduals = torch.sum(torch.square(yPred - testOutput))
# sumOfSquaresTotal = torch.sum(torch.square(testOutput - torch.mean(testOutput)))
# r2 = 1 - (sumOfSquaresOfResiduals / sumOfSquaresTotal)
#
# print(f"MAE {mae}")
# print(f"r2 {r2}")

# Final test accuracy

threshold = 0.5  # Adjust the threshold based on your requirements

# Convert predicted probabilities to binary predictions
binary_predictions = (yPred >= threshold).int()

# Convert binary_predictions to float
binary_predictions = binary_predictions.float().view(-1)
testOutput = testOutput.view(-1)

# print(binary_predictions)
# print(testOutput)

# Calculate accuracy
accuracy = (binary_predictions == testOutput).float().mean()

print(f"Accuracy: {accuracy.item() * 100}")

# Plot test accuracy
plt.plot(np.arange(1, numEpochs+1), testAccuracies)
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.ylim(0, 100)
plt.show()

# plot test loss
plt.plot(np.arange(1, numEpochs+1), testLosses)
plt.title("Test Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Loss")
plt.ylim(0, 1)
plt.show()