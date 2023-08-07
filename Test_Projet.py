# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:32:33 2023

@author: dioum
"""

import os
import sys
from glob import glob
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from itertools import combinations
import bisect
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster
from sklearn.cluster import  KMeans
from numpy.linalg import inv, qr,pinv
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import NMF
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import warnings filter
from warnings import simplefilter
import builtins
from sklearn.utils import resample
import fastcluster as fc
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet
import builtins
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import classification_report
# ignore all future warnings
import sklearn.metrics as metrics
from scipy.stats import pearsonr
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
simplefilter(action='ignore', category=FutureWarning)
#from iteration_utils import duplicates
import warnings
warnings.filterwarnings("ignore")



### Importation des données#####
meta_sample = pd.read_csv("meta.tsv",sep = "\t")
sample = pd.read_csv("all_samples.sat",sep = "\t")

## recuperation de la matrice d'abondance  ####

sample_data = sample.drop(['msp_name'],axis = 1)
name_sample = list(sample_data.columns)
name_sample.sort()
sample_data_ord = sample_data[name_sample]
name_msp = list(sample["msp_name"].values)
sample_data_ord_mat = sample_data_ord.T.to_numpy()

sample_data_ret = pd.DataFrame(sample_data_ord_mat ,columns = name_msp)
#C'est pour trouver l'echantillon de trop ##
diff_sample = list(set(sample_data_ord.columns).symmetric_difference(set(meta_sample["biosample_accession"])))

## Recherche de l'indice l'echatillon pour pouvoir le restirer ##
name = meta_sample["biosample_accession"].values
for i, elt in enumerate(name):
    if elt == 'SAMEA2466920':
        ind = i
print(ind) # ici ind = 31
 

## Recuperation de la variable d'interet (diagnostic) ###
our_target = meta_sample["diagnosis"]

##retrait de l'echantillon dans target ####
our_target_ret = our_target.drop(our_target.index[[ind]])
our_target_ret_cod = our_target_ret.copy()

## recoder notre variable d'interet en cancer et normal ########
for i in range(len(our_target_ret_cod)):
    if our_target_ret_cod.iloc[i] == "Small adenoma":
        our_target_ret_cod.iloc[i] = "Normal"
    elif our_target_ret_cod.iloc[i] == "Large adenoma":
        our_target_ret_cod.iloc[i] = "Cancer"
  
#our_target_ret_cod_bin = np.where(our_target_ret_cod == "Cancer",1,0)

# frequence de chaque modalité #########
our_target_ret.value_counts()

### ajout de la colone target recodée dans sample data #######
sample_data_ret_merg = sample_data_ret.copy() 
sample_data_ret_merg["diagnos"] = list(our_target_ret_cod)

#######Split notre table en train et test ###########
dfTrain, dfTest = train_test_split(sample_data_ret_merg,test_size=50,random_state=1,stratify=sample_data_ret_merg.diagnos)

####### Definir Arbre de decision et son ajustement ####
from sklearn.tree import DecisionTreeClassifier
arbreFirst = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=3)
arbreFirst.fit(X = dfTrain.iloc[:,:-1], y = dfTrain.diagnos)

## Arbre de classification avec les neouds############
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(arbreFirst,feature_names = list(sample_data_ret_merg.columns[:-1]),filled=True)

########## visualisation plus lisible ###############
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(arbreFirst,feature_names = list(sample_data_ret_merg.columns[:-1]),filled=True)
plt.show()


from sklearn.tree import export_text
tree_rules = export_text(arbreFirst,feature_names = list(sample_data_ret_merg.columns[:-1]),show_weights=True)


#  importance des variables sous forme d'un dataframe ########
impVarFirst={"Variable":sample_data_ret_merg.columns[:-1],"Importance":arbreFirst.feature_importances_}
print(pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False))

importances = arbreFirst.feature_importances_
indices = np.argsort(importances)[::-1]


colname = list(sample_data_ret.columns)

########### visualiasation des variables d'importances ############
plt.figure(1)
#plt.clf()
plt.title("Feature importances (10first)")
plt.barh(range(10), importances[indices][:10],
       color="r", align="center")
plt.yticks(range(10), [colname[i] for i in indices[:10]])
plt.show()

#prédiction sur l'échantillon test
predFirst = arbreFirst.predict(X=dfTest.iloc[:,:-1])
#distribution des predictions
print(np.unique(predFirst,return_counts=True))

# Standariser les données  pour une meilleur vision des courbes de ROC#
def stdise(X):
  mk=np.mean(X,axis=0)
  # Calcul de l'écart-type avec max pour éviter une division par 0
  sk=np.maximum(np.std(X,axis=0),10*np.finfo(float).eps)
  Xs=np.add(X,-mk)
  Xs=np.multiply(Xs,1/sk)
  return Xs

X = stdise(sample_data_ret)/np.sqrt(np.shape(sample_data_ret)[0])



def ROC(y_test,y_score,methodName=" ",plot=True):

    ntest = np.size(y_test,0)
    B = np.size(y_test,1)
    fpr, tpr, _ = roc_curve(np.reshape(y_test,B*ntest), np.reshape(y_score,B*ntest))
#    if len(fpr)<3:
#        print("Problem: len(fpr) is lower than 3")
#        return
    roc_auc = auc(fpr, tpr)

    if plot:
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(methodName)
        plt.legend(loc="lower right")
        plt.show()
    return(roc_auc)


########conversion de la variable cible en binaire ###########
our_target_ret_cod_bin = our_target_ret_cod.copy()
our_target_ret_cod_bin =  np.where(our_target_ret_cod_bin == "Cancer",1,0)


########## Decision Tree ##################
B = 1981#185
n_test = 50
y_score = np.zeros([n_test,B])
y_test_all = np.zeros([n_test,B])
for b in range(B):
    X_train, X_test, y_train, y_test = train_test_split(X,our_target_ret_cod_bin,test_size=n_test)
    mk=np.mean(X_train,axis=0)
    sk=np.maximum(np.std(X_train,axis=0),10*np.finfo(float).eps)
    X_train, X_test = np.add(X_train,-mk), np.add(X_test,-mk)
    X_train, X_test = np.multiply(X_train,1/sk),np.multiply(X_test,1/sk)
    clf =  DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_score[:,b] = clf.predict_proba(X_test)[:,1]
    y_test_all[:,b] = y_test

ROC(y_test_all,y_score,"Tree")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]



# 
plt.figure(1)
plt.clf()
plt.title("Feature importances (20 first)")
plt.barh(range(20), importances[indices][:20],
       color="r", align="center")
plt.yticks(range(20), [colname[i] for i in indices[:20]])


######################### Random Forest ########################


from sklearn.ensemble import RandomForestClassifier

B = 1981#185
n_test = 50
y_score = np.zeros([n_test,B])
y_test_all = np.zeros([n_test,B])
for b in range(B):
    X_train, X_test, y_train, y_test = train_test_split(X,our_target_ret_cod_bin,test_size=n_test)
    mk=np.mean(X_train,axis=0)
    sk=np.maximum(np.std(X_train,axis=0),10*np.finfo(float).eps)
    X_train, X_test = np.add(X_train,-mk), np.add(X_test,-mk)
    X_train, X_test = np.multiply(X_train,1/sk),np.multiply(X_test,1/sk)
    clf = RandomForestClassifier(n_estimators=20,max_depth=5)
    clf.fit(X_train,y_train)
    y_score[:,b] = clf.predict_proba(X_test)[:,1]
    y_test_all[:,b] = y_test

ROC(y_test_all,y_score,"Random Forest")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(1)
plt.clf()
plt.title("Feature importances (20 first)")
plt.barh(range(20), importances[indices][:20],
       color="r", align="center")
plt.yticks(range(20), [colname[i] for i in indices[:20]])


############ Boosting ##############

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


B = 1981#185
n_test = 50
y_score = np.zeros([n_test,B])
y_test_all = np.zeros([n_test,B])
for b in range(B):
    X_train, X_test, y_train, y_test = train_test_split(X,our_target_ret_cod_bin,test_size=n_test)
    mk=np.mean(X_train,axis=0)
    sk=np.maximum(np.std(X_train,axis=0),10*np.finfo(float).eps)
    X_train, X_test = np.add(X_train,-mk), np.add(X_test,-mk)
    X_train, X_test = np.multiply(X_train,1/sk),np.multiply(X_test,1/sk)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=20)
    clf.fit(X_train,y_train)
    y_score[:,b] = clf.predict_proba(X_test)[:,1]
    y_test_all[:,b] = y_test

ROC(y_test_all,y_score,"AdaBoost")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(1)
plt.clf()
plt.title("Feature importances (20 first)")
plt.barh(range(20), importances[indices][:20],
       color="r", align="center")
plt.yticks(range(20), [colname[i] for i in indices[:20]])

##################Gradient Boosting ##################

from sklearn.ensemble import GradientBoostingClassifier

B =  1981 #1981
n_test = 50
y_score = np.zeros([n_test,B])
y_test_all = np.zeros([n_test,B])
for b in range(B):
    X_train, X_test, y_train, y_test = train_test_split(X,our_target_ret_cod_bin,test_size=n_test)
    mk=np.mean(X_train,axis=0)
    sk=np.maximum(np.std(X_train,axis=0),10*np.finfo(float).eps)
    X_train, X_test = np.add(X_train,-mk), np.add(X_test,-mk)
    X_train, X_test = np.multiply(X_train,1/sk),np.multiply(X_test,1/sk)
    clf = GradientBoostingClassifier(loss='exponential',n_estimators=200,max_features=5)
    clf.fit(X_train,y_train)
    y_score[:,b] = clf.predict_proba(X_test)[:,1]
    y_test_all[:,b] = y_test

ROC(y_test_all,y_score,"Gradient Boosting")
importances = clf.feature_importances_
indices_boost = np.argsort(importances)[::-1]

plt.figure(1)
plt.clf()
plt.title("Feature importances (20 first)")
plt.barh(range(20), importances[indices_boost][:20],
       color="r", align="center")
plt.yticks(range(20), [colname[i] for i in indices_boost[:20]])


####### table variable d'importance #######

#importance des variables
impVarFirst={"Variable":sample_data_ret_merg.columns[:-1],"Importance":importances}
esp_tab = pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False)

esp_tab_mod = esp_tab[esp_tab['Importance'] > 0]

sample_gr = sample.iloc[esp_tab_mod.index,:]
sample_gr.to_csv(r' sample_gr.csv', index = False)

sample_gr_map = sample_gr.copy()
sample_gr_map_red = sample_gr_map.drop(["msp_name"],axis = 1)

#### split Data ####

data_train, data_test, target_train, target_test = train_test_split(
    sample_data_ret, our_target_ret_cod, random_state=0)
range_features = {
    feature_name: (sample_data_ret[feature_name].min() - 1, sample_data_ret[feature_name].max() + 1)
    for feature_name in sample_data_ret.columns}

#Prédiction: arbre de decision #########
     
#prédiction sur l'échantillon test

arbreFirst =  DecisionTreeClassifier()
arbreFirst.fit(data_train,target_train)
predFirst = arbreFirst.predict(X= data_test)

#distribution des predictions
print(np.unique(predFirst,return_counts=True))

#rapport de prédiction
print(metrics.classification_report(target_test,predFirst))


#Prédiction: Random Forest #########
RandomF = RandomForestClassifier(n_estimators=20,max_depth=5)
RandomF.fit(data_train,target_train)
predFirst = RandomF.predict(X= data_test)

#distribution des predictions
print(np.unique(predFirst,return_counts=True))

#rapport de prédiction
print(metrics.classification_report(target_test,predFirst))



#Prédiction: AdaBoosting #########
AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=20)
AdaBoost.fit(data_train,target_train)
predFirst = AdaBoost.predict(X= data_test)

#distribution des predictions
print(np.unique(predFirst,return_counts=True))

#rapport de prédiction
print(metrics.classification_report(target_test,predFirst))


#Prédiction: GradienBoosting #########
GradBoost = GradientBoostingClassifier(loss='exponential',n_estimators=200,max_features=5)
GradBoost.fit(data_train,target_train)
predFirst = GradBoost.predict(X= data_test)

#distribution des predictions
print(np.unique(predFirst,return_counts=True))

#rapport de prédiction
print(metrics.classification_report(target_test,predFirst))