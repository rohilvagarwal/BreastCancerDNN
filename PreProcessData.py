import numpy as np
import torch

#np.set_printoptions(threshold=np.inf, linewidth=400)

class PreProcessData:
	def __init__(self, data: np.ndarray):
		self.originalData = data
		self.preprocessedData = data

		self.all_preprocessing_steps()

		self.preprocessedData = self.preprocessedData.astype(float)

		self.trainingData = None
		self.testingData = None

		self.split_train_and_test()


	def split_train_and_test(self):
		#80% training 20% testing
		amtTraining = int(len(self.preprocessedData) * 4 / 5)

		#Training data is 4/5 of total data - will separate into input and output in dataloader
		self.trainingData = self.preprocessedData[:amtTraining, :]
		self.testingData = self.preprocessedData[amtTraining:, :]

	def get_preprocessed_data(self):
		return self.preprocessedData

	def remove_unneeded_columns(self):
		self.preprocessedData = self.preprocessedData[:, 1:-1]

	def convert_categorical_data(self):
		column_index = 0  # Replace with the index of the desired column
		column_values = self.preprocessedData[:, column_index]  # Extract the column values
		unique_values = np.unique(column_values)  # Get the unique values in the column
		value_mapping = dict(zip(unique_values, range(len(unique_values))))  # Create a mapping of unique values to numbers
		converted_values = np.array([value_mapping[value] for value in column_values])  # Convert the column values to numbers
		self.preprocessedData[:, column_index] = converted_values  # Replace the column values with the converted numbers

	def move_dependent_variable(self):
		#move dependent variable to end
		columnOrderArray = []
		dependentVarColumn = 0
		for x in range(self.preprocessedData.shape[1]):
			if x != dependentVarColumn:
				columnOrderArray.append(x)

		columnOrderArray.append(dependentVarColumn)

		self.preprocessedData = self.preprocessedData[:, columnOrderArray]

	def all_preprocessing_steps(self):
		self.remove_unneeded_columns()
		self.convert_categorical_data()
		self.move_dependent_variable()

	def get_train_and_test(self):
		return self.trainingData, self.testingData