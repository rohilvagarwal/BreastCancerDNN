import numpy as np

# get data
data = np.loadtxt(fname='breast_cancer_data/wdbc.data', dtype=str, delimiter=',')

# remove id column
data = data[:, 1:-1]


