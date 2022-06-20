#Heart Disease

#Imports
import pandas as pd
import seaborn as sns

from keras import models
from keras import layers

from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical  ---> Sometime makes ERRORS
import tensorflow as tf
from tensorflow.keras.models import load_model     # loading the model
from tensorflow.keras.models import model_from_json  # loading the model architecture

import matplotlib.pyplot as plt
import numpy as np

# for confusion matrix (old way)
import itertools
from sklearn.metrics import confusion_matrix

# for confusin matrix (new way)
from sklearn.metrics import plot_confusion_matrix



#Read the dataset
df = pd.read_csv("heart.csv")


#Data Visualization

#Graph_1
def countplot_1(df):
    plt.ion()
    plt.figure(figsize=(12,5))
    f = sns.countplot(x='target', data=df)
    f.set_title("Heart disease distribution")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    plt.pause(2)
    plt.show()


#Graph_2
def countplot_2(df):
    plt.ion()
    plt.figure(figsize=(12,5))
    f = sns.countplot(x='target', data=df, hue='sex')
    plt.legend(['Female', 'Male'])
    f.set_title("Heart disease by gender")
    f.set_xticklabels(['No Heart disease', 'Heart Disease'])
    plt.xlabel("")
    plt.pause(2)
    plt.show()


#Graph_3
def countplot_3(df):
    plt.ion()
    plt.figure(figsize=(12,8))
    heat_map = sns.heatmap(df.corr(method='pearson'), annot=True,
    fmt='.2f', linewidths=2)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
    plt.rcParams["figure.figsize"] = (50,50)
    plt.pause(2)
    plt.show()


#Calls of graphs
countplot_1(df)
countplot_2(df)
countplot_3(df)




# Correlation between target and other features
print(df.corr()["target"].abs().sort_values(ascending=False))


#Dropping undeeded columns from the data
df = df.drop(["chol", "fbs"],  axis=1)


#Scaling the data
df["age"] = df["age"]/df["age"].max()

df["trestbps"] = df["trestbps"]/df["trestbps"].max()

# df["chol"] = df["chol"]/df["chol"].max()

df["thalach"] = df["thalach"]/df["thalach"].max()

df["oldpeak"] = df["oldpeak"]/df["oldpeak"].max()

df["ca"] = df["ca"]/df["ca"].max()


#Dummies
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

X = X.to_numpy()
y = y.to_numpy()


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print("Xtrain shape: ", X_train.shape)
print("ytrain shape: ", y_train.shape)
print("Xtest shape: ", X_test.shape)
print("ytest shape: ", y_test.shape)


#Parameters
epoch = 3
activ = ["sigmoid", "relu", "softmax"]

# Hidden layers
layer_1 = 13 # Number of Neurons
layer_2 = 9



# Calling the module
from NN_fit_model_module import Fit_model
# Instance
fit = Fit_model(layer_1, layer_2, activ, X, epoch, X_train, y_train)





#Plot
def plotting(history):
    plt.ion()
    plt.figure(figsize=(12,8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training_acc", "Validation_acc"])
    plt.pause(2)
    plt.show()


#Call of plot
plotting(fit.history)




# Test score for test data using evaluation
test1_score, test2_score = fit.network.evaluate(X_test, y_test)

print('Test Loss:', test1_score)
print('Test Accuracy: ', test2_score)


#Prediction
prediction = fit.network.predict(x = X_test)


#Rounded Predictions
rounded_prediction = np.round(prediction)




#Confusion Matrix
# 1. Variant
# Code is copied from Scikit website

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.ion()
plt.figure(figsize=(12,5))
cm = confusion_matrix(y_true=y_test, y_pred=rounded_prediction)

# Create labels for my plot --> Classes
cm_plot_labels = ["0", "1"]

plot_confusion_matrix(cm= cm, classes= cm_plot_labels, title= "Confusion matrix")
plt.pause(2)
plt.show()



# Save Model


fit.network.save("./models/heart_disease.h5")

fit.network.save_weights("./models/heart_disease_weights.h5")


# Save the architecture to json string
json_string = fit.network.to_json()

# Save the architecture to YAML string  --> Removed due to security risks
#yaml_string = network.to_yaml()

print(json_string)



#Load the model

# Loading the whole Model:

new_model = load_model("./models/heart_disease.h5")

# Gets the summary of the loaded mode
new_model.summary()

# Gets the optimizer name
print("My Optimizer is: " , new_model.optimizer)


# Get the weights
new_model.get_weights()



# Load the weight

# Create a new model
model_2 = models.Sequential(
    [
        layers.Dense(layer_1, activation=activ[0], input_shape=(X.shape[1],)),
        layers.Dense(layer_2, activation=activ[0]),
        layers.Dense(1, activation=activ[0])
    ]
)

# Model2 Compile (Config)


# Load Weights
model_2.load_weights("./models/heart_disease_weights.h5")



# Loading the architecture

model_architecture = model_from_json(json_string)

model_architecture.summary()



model_2.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


history_2 = model_2.fit(X_train, y_train, epochs=epoch, shuffle=True, verbose=1, validation_split= 0.1, batch_size=5)


prediction = model_2.predict(x = X_test)


rounded_prediction = np.round(prediction)


# 1. Variant
# Code is copied from Scikit website

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.ion()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.pause(2)




plt.figure(figsize=(12,5))
cm = confusion_matrix(y_true=y_test, y_pred=rounded_prediction)

# Create labels for my plot --> Classes
cm_plot_labels = ["0", "1"]

plot_confusion_matrix(cm= cm, classes= cm_plot_labels, title= "Confusion matrix")