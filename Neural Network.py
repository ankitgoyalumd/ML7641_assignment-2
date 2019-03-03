# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:16:45 2019

@author: Ankit Goyal
"""

import mlrose
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.legend_handler import HandlerLine2D
from sklearn import tree
import numpy as np
import os
import time

os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1')
data=pd.read_csv('weather_data.csv')
data=data.dropna()
os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 1/Dataset/nba')
data=pd.read_csv('nba.csv')
data=data.dropna()
print(data.info())
fields_to_drop=['TARGET_5Yrs','Name']
X_2 = data.drop(fields_to_drop,axis=1)
y_2 = data['TARGET_5Yrs']
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.20)


#Mapping RainTomoorow and Rain Today to 0 and 1
data['RainToday'] = data['RainToday'].map({'No': 0, 'Yes': 1})
data['RainTomorrow'] = data['RainTomorrow'].map({'No': 0, 'Yes': 1})


#Adding dummy variable for attributes with more than 2 categories
Categorical= ['WindGustDir', 'WindDir9am', 'WindDir3pm']
for each in Categorical:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)
df = pd.concat([data, dummies], axis=1)
fields_to_drop = ['Date', 'Location','WindGustDir', 'WindDir9am', 'WindDir3pm']
df = df.drop(fields_to_drop, axis=1)

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

schedule=mlrose.GeomDecay(init_temp=10000)
lr=[0.2,0.3,0.4,0.5,0.6,0.7,0.9]
def learn_rate(lr):
    test_accuracy=[]
    train_accuracy=[]
    learning_rate=[]
    for c in lr:
        #schedule=mlrose.GeomDecay(init_temp=c)
        nn_model1=mlrose.NeuralNetwork(hidden_nodes=[4,4,4],activation='relu',algorithm='genetic_alg',max_iters=1000,mutation_prob=0.4,bias=True,learning_rate=0.01,pop_size=200,is_classifier=True,early_stopping=True,clip_max=5,max_attempts=100)
        nn_model1.fit(X_train,y_train)
        y_train_pred=nn_model1.predict(X_train)
        y_test_pred=nn_model1.predict(X_test)
        y_train_accuracy=accuracy_score(y_train,y_train_pred)
        y_test_accuracy=accuracy_score(y_test,y_test_pred)
        test_accuracy.append(y_test_accuracy)
        train_accuracy.append(y_train_accuracy)
        learning_rate.append(c)
        print(test_accuracy)
        #print(nn_model1.fitted_weights)
    train_accuracy=np.asarray(train_accuracy)
    test_accuracy=np.asarray(test_accuracy)
    learning_rate=np.asarray(learning_rate)
    line1, = plt.plot(learning_rate,train_accuracy,color='r',label='train_accuracy_uniform')
    line2, = plt.plot(learning_rate,test_accuracy,color='b',label='test_accuracy_uniform')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('mutation probability')
    return None
#learn_rate(lr)
lr=[0.1,0.2,0.3,0.4,0.5,1]
def iterations(lr):
    test_accuracy=[]
    train_accuracy=[]
    iters=[]
    #schedule=mlrose.GeomDecay()
    for c in lr:
        nn_model1=mlrose.NeuralNetwork(hidden_nodes=[2],activation='relu',algorithm='genetic_alg',mutation_prob=c,max_iters=100,bias=True,learning_rate=0.01,is_classifier=True,early_stopping=True,max_attempts=100,clip_max=5)
        nn_model1.fit(X_train,y_train)
        y_train_pred=nn_model1.predict(X_train)
        y_test_pred=nn_model1.predict(X_test)
        y_train_accuracy=accuracy_score(y_train,y_train_pred)
        y_test_accuracy=accuracy_score(y_test,y_test_pred)
        test_accuracy.append(y_test_accuracy)
        train_accuracy.append(y_train_accuracy)
        iters.append(c)
        print(test_accuracy)
        print(train_accuracy)
    train_accuracy=np.asarray(train_accuracy)
    test_accuracy=np.asarray(test_accuracy)
    iters=np.asarray(iters)
    line1, = plt.plot(iters,train_accuracy,color='r',label='train_accuracy_uniform')
    line2, = plt.plot(iters,test_accuracy,color='b',label='test_accuracy_uniform')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Accuracy_score')
    plt.xlabel('number of iterations')
    return None
#iterations(lr)
schedule=mlrose.GeomDecay(init_temp=10000)
nn_model1=mlrose.NeuralNetwork(hidden_nodes=[4,4,4],activation='relu',algorithm='random_hill_climb',max_iters=1000,bias=True,learning_rate=0.1,is_classifier=True,early_stopping=True,max_attempts=100,clip_max=5)

start=time.time()
nn_model1.fit(X_train,y_train)
print('time is',time.time()-start)
y_train_pred=nn_model1.predict(X_train)
y_test_pred=nn_model1.predict(X_test)
y_train_accuracy=accuracy_score(y_train,y_train_pred)
y_test_accuracy=accuracy_score(y_test,y_test_pred)
print(y_test_accuracy)
print(y_train_accuracy)

#iterations_gen()