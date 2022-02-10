# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:36:28 2022

@author: theo-
"""


import sklearn

import tqdm
import h5py
import pandas as pd
import numpy as np



# recuperation des données

    
titanic_train = pd.read_csv('dataset_titanic/titanic_train.csv')
titanic_test = pd.read_csv('dataset_titanic/titanic_test.csv')


# triage des donnée interessante pour la classification

titanic_train = titanic_train[['survived', 'pclass','sex', 'age']]


print(titanic_test.head(5))

titanic_test = titanic_test[[ 'pclass','sex', 'age']]

# normaliser transformer les string en 0 ou 1 :
    
titanic_train['sex'].replace(['male','female'], [0,1], inplace = True)
titanic_test['sex'].replace(['male','female'], [0,1], inplace = True)


titanic_train.dropna(axis=0, inplace=True)




# DEVELOPPEMENT MODELE DE CLASSIFICATION
# ON UTILISE KNEIGBORS CLASSIFER



from sklearn.neighbors import KNeighborsClassifier


y = titanic_train['survived']


X = titanic_train[['pclass', 'sex', 'age']]


model = KNeighborsClassifier(n_neighbors=2)



#entrainement
def entrainement (X,y, model):
    model.fit(X,y)    
    print(model.score(X,y))




#prediction score

def predict (X, model):   
    model.predict(X)
    print(model.predict(X))



def survie(model, pclass = 3, sex = 0, age = 26):
    
    x = np.array([pclass, sex, age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))



"""
for i in range (1,100):
    model = KNeighborsClassifier(n_neighbors=i)
    print("pour i = ",i)
    entrainement(X, y, model)
    
    """
    
    
# on remarque que le modele predit au mieux pour i = 1

