import torch
import numpy as np
import matplotlib.pyplot as plt
from PreProcessData import PreProcessData

data = np.loadtxt(fname='breast_cancer_data/wdbc.data', dtype=str, delimiter=',')  # load data

preprocessedData = PreProcessData(data)  # preprocess data

data = preprocessedData.get_preprocessed_data().T  # load preprocessed dataset with rows as variables and columns as values

corr_matrix = torch.corrcoef(torch.tensor(data))  # calculate correlations matrix

plt.imshow(corr_matrix, cmap="GnBu")  # show matrix as image
plt.colorbar()  # show color bar (essentially legend)
plt.title("Heatmap of data category correlations")
plt.show()  # show plot
