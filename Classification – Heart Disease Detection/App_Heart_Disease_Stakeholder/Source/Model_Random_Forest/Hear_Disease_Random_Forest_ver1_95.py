#Heart Disease Model with Random Forest

#Imports
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras import models
from keras import layers

from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical  ---> Sometime makes ERRORS
import tensorflow as tf
from tensorflow.keras.models import load_model     # loading the model
from tensorflow.keras.models import model_from_json  # loading the model architecture


import numpy as np

# for confusion matrix (old way)
import itertools
from sklearn.metrics import confusion_matrix

# for confusin matrix (new way)



#Read the data
df = pd.read_csv("heart.csv")



#Data Visualization
#1st Graph
def countplot_1(df):
    plt.ion()
    plt.figure(figsize=(12,5))
    f = sns.countplot(x='target', data=df)
    f.set_title("Heart disease distribution")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    
    plt.show()
    plt.pause(2)

#2nd Graph
def countplot_2(df):
    plt.ion()
    plt.figure(figsize=(12,5))
    f = sns.countplot(x='target', data=df, hue='sex')
    plt.legend(['Female', 'Male'])
    f.set_title("Heart disease by gender")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    
    plt.show()
    plt.pause(2)



#3rd Graph
def countplot_3(df):
    plt.ion()
    plt.figure(figsize=(12,8))
    heat_map = sns.heatmap(df.corr(method='pearson'), annot=True,
    fmt='.2f', linewidths=2)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
    plt.rcParams["figure.figsize"] = (50,50)
    
    plt.show()
    plt.pause(2)


# Calls of graphs
countplot_1(df)
countplot_2(df)
countplot_3(df)



# Correlation between target and other features
print(df.corr()["target"].abs().sort_values(ascending=False))


#Dropping undeeded columns from the data
df = df.drop(["chol", "fbs"],  axis=1)


#Encoding
dummies_1 = pd.get_dummies(df.cp, prefix='cp')
df = pd.concat([df, dummies_1],axis='columns')
dummies_2 = pd.get_dummies(df.thal, prefix='thal')
df = pd.concat([df, dummies_2],axis='columns')
dummies_3 = pd.get_dummies(df.slope, prefix='slope')
df = pd.concat([df, dummies_3],axis='columns')


df = df.drop(['cp', 'thal', 'slope'], axis='columns')


#Defining X (input) and y (target)
X = df.drop(['target'], axis=1)
y = df['target']


#Training the model with train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("Xtrain shape: ", X_train.shape)
print("ytrain shape: ", y_train.shape)
print("Xtest shape: ", X_test.shape)
print("ytest shape: ", y_test.shape)




# Calling the module
from random_forest_module import Train_selection

# Instance
train_model = Train_selection(X_train, y_train,X_test, y_test)



# Confusion Matrix
def confusion_matrix(y_test, y_predicted):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_predicted) # Truth value, prediction

    import seaborn as sn
    plt.ion()
    plt.figure(figsize=(10,7))
    sn.heatmap(cm,annot=True)
    plt.xlabel('y_predicted')
    plt.ylabel('y_test (Truth)')
    
    plt.show()
    plt.pause(2)


# Call of matrix
confusion_matrix(y_test, train_model.y_predicted)


# Save model
import pickle # this module allows to serialize your model into a file

with open('model_pickle','wb') as file: # first open a file write a binary
    pickle.dump(train_model.model,file)