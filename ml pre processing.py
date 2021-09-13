# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('Data.csv')
X=data.iloc[:,:3].values
y=data.iloc[:,-1:].values


from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values =np.nan,strategy="mean")
X[:,1:3]=np.nan_to_num(X[:,1:3])
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_x =LabelEncoder()
X[:,0]=LabelEncoder_x.fit_transform(X[:,0])

LabelEncoder_y =LabelEncoder()
y=LabelEncoder_x.fit_transform(y)

'''oneHotEncoder=OneHotEncoder(categories=[0])
X[:,0]=OneHotEncoder.fit_transform(X[:,0]).toarray()'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

 