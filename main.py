# This is an exercise project for DT and their ensambles, using iris flower dataset and california housing dataset
# importing libraries
import numpy as np
from sklearn import datasets
import seaborn as sbn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Random seed
seed = 100
np.random.seed(seed)
#get the data
iris = datasets.load_iris()
data = pd.DataFrame({'sepal_length': iris.data[:,0],
                   'sepal_width': iris.data[:,1],
                   'petal_length': iris.data[:,2],
                   'petal_width': iris.data[:,3],
                   'type': iris.target})
print(data)
#split using sklearn

X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2) # random_state for reproducible output
print(f'x_tr: {X_train.shape} ; y_tr: {Y_train.shape} ; x_tst: {X_test.shape} ; y_tst: {Y_test.shape}')#print shape of splits