# pre processing data set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set
dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Splitting the dataset into trainning set and testing set
from sklearn.model_selection import train_test_split as tts
X_train , X_test, Y_train , Y_test = tts(X , Y , test_size = 0.2 , random_state = 0)
# test_size-how many data set should be marked for testing here 20%

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler 
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
# You dont have to fit here cause you have already fitted the way it is in line 36
X_test= sc_X.transform(X_test)'''






