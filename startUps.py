from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


dataset = pd.read_csv('50start.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
dataset.head()

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categories =[3])
X = onehotencoder.fit_transform(X).toarray()