# Analysis

## PreProcessData.py
Preprocesses data for usage

__init__ ==> constructor(data = dataset : 2d numpy array)  returns None:
Makes calls to preprocess data

__split_train_and_test__ ==> splits data into 80:20 ratio for training and testing, respectively

__get_preprocessed_data__ ==> getter function for preprocessed data

__remove_unneeded_columns__ ==> remove irrelevant columns

__convert_categorical_data__ ==> convert all categorical data to numeric data (only the target column, in this case)

__move_dependent_variable__ ==> moves dependent variable to the last column

__all_preprocessing_steps__ ==> calls the preprocessing steps

__get_train_and_test__ ==> get training and testing data

## DNNDataLoader.py
Extends torch.utils.data.Dataset to provide functionality for DataLoader

__init__ ==> constructor(df = dataset : 2d numpy array) returns None:  
Splits dataset into input and output tensors of data type float, with the output being the last column of the original dataset  

__len__ ==> enables the length function's use on the class

__getitem__ ==> enables indexing for the dataset, returning (input, output)

## NeuralNetworkNet.py
Creates neural network model by extending torch.nn.Module

__init__ ==> constructor creates the following layers  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear(29, 64) with ReLU activation function  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear(64, 128) with ReLU activation function  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear(128, 128) with ReLU activation function  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear(128, 32) with ReLU activation function  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Linear(32, 1) with Sigmoid activation function (**output**)  

__forward__ ==> implements torch.nn.Module forward method that enables model training and testing -- simply uses the layers in sequence

## correlations.py
Maps out correlations between data categories using torch.corrcoef() and matplotlib.pyplot.imshow(0

## main.py
Main function for running the program
1. Import necessary packages and classes
2. Import data
3. Preprocess data with PreProcessData class
4. Load data with DNNDataLoader class
5. Separate test input and output
6. Set epoch and batch size
7. Train model
8. Test model after each epoch and append accuracy to list
9. Calculate and output final model accuracy with test dataset
10. Graph model accuracy over epochs using the list from Step 8
