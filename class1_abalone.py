# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:51:54 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
abalone= pd.read_csv('D:/ML_Senthil/abalone.csv')
abalone.head()

targ= abalone['Rings']
targ.head()

#inplace = True will update the dataframe after droping the columns means we
#do not need to store it an any other variable instead we can place inplace= True
#for ordinal data go for level encoding
#for nominal data go for dummies if class is greater than 2

#here rings is the target variable because with the rings we can predict the weight
abalone.drop(['Rings'], axis=1, inplace= True)

gen= pd.get_dummies(abalone)
gen.head()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
[xtrain, xval, ytrain, yval]= train_test_split(gen, targ, test_size= 0.3, random_state=100)
regmod= LinearRegression()
regmod.fit(xtrain, ytrain)
ypred= regmod.predict(xval)

mse= mean_squared_error(yval, ypred)
rmse= np.sqrt(mse)
#xval is input column, yval is output column
rsq= regmod.score(xval, yval)


#overfitting is when in real case scenario the accuracy decreases bcz while checking for accuracy
#we had done hyperparameter tuning and over fitted the model so, 