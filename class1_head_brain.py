# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:28:36 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
head_brain= pd.read_csv('C:/Users/Administrator/Downloads/headbrain.csv')
head_brain.head()

head_brain= head_brain.drop(['Gender', 'Age Range'], axis=1)

#values is used to convert the dataframe to array format so that numpy operations can be performed.
x= head_brain['Head Size(cm^3)'].values
X= x.reshape(-1,1)

y= head_brain['Brain Weight(grams)'].values
Y= y.reshape(-1,1)

x_mean= X.mean()
y_mean= np.mean(Y)

b1= np.sum((X-x_mean)*(Y-y_mean))/ np.sum(np.square(X- x_mean))

b0= y_mean- b1*x_mean

y_pred= b1*X + b0

y_pred= y_pred.reshape(-1,1)
out= np.concatenate((Y, y_pred), axis=1)

res= sum(Y-y_pred)
sq_err= sum((Y-y_pred)**2)
mse= np.mean((Y-y_pred)**2)
rmse=  np.sqrt(np.mean((Y-y_pred)**2))

print("The residue is:")
print(res)

print("The squared error is:")
print(sq_err)

print("The mean squared error is:")
print(mse)

print("The root squared error is:")
print(rmse)

R_square= 1- np.sum((Y- y_pred)**2)/np.sum((Y- y_mean)**2)
R_square

R_adjusted= (1-R_square)*(5-1)/ (5-1-1)


