import numpy as np

# get data
data = np.loadtxt(fname='breast_cancer_data/wdbc.data', dtype=str, delimiter=',')

# remove id column
data = data[:, 1:-1]

# only first column is categorical
uniques = np.unique(data[:, 0])
dict_unique = dict(enumerate(uniques)) # number unique values
dict_unique = dict([(value, key) for key, value in dict_unique.items()])  # swap key and value pairs to make it (unique value, number)
data[:, 0] = [dict_unique[value] for value in data[:, 0]] # replace values with numbers
data = data.astype(float)

print(data)
