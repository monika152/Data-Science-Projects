# Is the salary more or less than 50K?


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Read the data
df = pd.read_csv('adult.csv')
df_original = pd.read_csv('adult.csv')

#Data Cleaning
df = df.replace("?", np.nan)
df = df.dropna(axis=0, how='any')


#Visualization 1st graph
def graph_1(df):

    plt.ion()
    plt.figure(figsize=(12,4))
    ax = sns.countplot(data = df, x = 'sex', hue="income", palette = 'rocket')

    plt.xlabel("Sex", fontsize= 12)
    plt.ylabel("# of People", fontsize= 12)
    plt.ylim(0,20000)
    plt.xticks([0,1],['Male', 'Female'], fontsize = 11)

    for p in ax.patches:
        ax.annotate((p.get_height()), (p.get_x()+0.16, p.get_height()+1000))

    plt.pause(2)
    plt.show()


# visualization 2nd graph
def graph_2(df):
    
    plt.ion()
    plt.figure(figsize=(12,4))
    edu = df["education"].value_counts(normalize=True)

    sns.barplot(edu.values, edu.index, palette='mako')
    plt.title('Education')
    plt.xlabel('Number of people')
    plt.ylabel('Education vs Number of people')
    plt.tick_params(labelsize=12)

    plt.pause(2)
    plt.show()



#Visualization 3rd graph
def graph_3(df):

    plt.ion()
    plt.figure(figsize=(15,5))

    sns.distplot(df['hours.per.week'])
    plt.ticklabel_format(style='plain', axis='x') #repressing scientific notation on x
    plt.ylabel('')

    plt.pause(2)
    plt.show()




# Calls for graphs
graph_1(df)
graph_2(df)
graph_3(df)




#Data Encoding
dummies = pd.get_dummies(df.income)
print(dummies)

merged = pd.concat([df,dummies], axis='columns') # to add prefix: ..., prefix=["Quarter_"] )


final_data = merged.drop(['income'],axis='columns')
print(final_data)



#Splitting the data
final = final_data.drop(['<=50K'], axis='columns')

final = final.drop(["fnlwgt","education","marital.status","relationship","capital.loss","native.country"],axis=1)



#Label Encoding
from sklearn.preprocessing import LabelEncoder
le_workclass = LabelEncoder()
le_race = LabelEncoder()
le_sex = LabelEncoder()
le_occupation = LabelEncoder()


final['workclass'] = le_workclass.fit_transform(final['workclass'])
final['race'] = le_race.fit_transform(final['race'])
final['sex'] = le_sex.fit_transform(final['sex'])
final['occupation'] = le_sex.fit_transform(final['occupation'])
print(final)



#Defining X (input) and y (target)
X = final.drop([">50K"], axis = "columns")

target = final['>50K']
y = target



#Import Module

from training_module import Train_predict

train_predict = Train_predict(X, y)
train_predict.prediction()