#-*-coding:utf-8 -*-
import time
from numpy import *
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle

from pandas import Series,DataFrame

import pandas as pd
#set_printoptions(threshold='nan')
n=10
def dis_erro(pred_y,rel_y):
    diffMat = pred_y - rel_y
    sqDiff = diffMat ** 2
    sqDis = sqDiff.sum(1)
    dis_erro_values = sqDis ** 0.5
    ave_erro1 = mean((dis_erro_values).tolist())
    return ave_erro1
##############################################################################
features=[[0, 1, 2, 3, 4],[0, 1, 2, 3, 4, 13, 19],[0, 1, 2, 3, 4, 13, 19],[0, 1, 2, 3, 4, 13],[0, 1, 2, 3, 4, 13, 19],[5, 6, 8, 14, 15, 16, 17, 18],\
[5, 6, 7, 8, 9, 10, 11, 12, 15, 16],[6, 7, 8, 9, 10, 11, 12, 16],[5, 6, 7, 8, 9, 10, 11, 12, 16],[6, 7, 8, 9, 10, 11, 12],\
[6, 7, 8, 9, 10, 11, 12],[6, 7, 8, 9, 10, 11, 12],[6, 7, 8, 9, 10, 11, 12, 15, 16],[1, 2, 3, 4, 13, 18, 19],[5, 14, 15, 16, 17, 18, 19],\
[5, 6, 12, 14, 15, 16],[5, 6, 7, 8, 12, 14, 15, 16],[5, 14, 17, 18],[5, 13, 14, 17, 18, 19],[1, 2, 4, 13, 14, 18, 19]]
list_trans=[]
for i in features:
    full_features=zeros(21)
    full_features[i]=1
    list_trans.append(full_features)
list_trans=array(list_trans)
clf=KMeans(n_clusters=n,random_state=50,n_init=100)
pred_k=clf.fit_predict(list_trans)
c_kinds={}
for i,j in zip(pred_k,features):
    #print i,j
    if c_kinds.has_key(i):
        c_kinds[i]=c_kinds[i]+j
    else:
        c_kinds[i]=j
for i in c_kinds:
    c_kinds[i]=list(set(c_kinds[i]))
    print i,c_kinds[i]
#print clf.inertia_
#print clf.cluster_centers_
####################################################
start=time.time()
f = open('load_data2.pkl','r+')
data= pickle.load(f)

training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.02, random_state = 3)
#####################################################
training_model={}
for i in range(n):
    training_model[i] = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(training_x[:,c_kinds[i]],training_y)
a=[]
b=[]
num=0
for i,k in zip(test_x,test_y):
    full_features=zeros(21)
    full_features[i!=100]=1
    full_features=full_features.reshape(1,-1)
    kind=clf.predict(full_features)
    kind=kind[0]
    index=c_kinds[kind]
    t=i[index].reshape(1,-1)
    pred_y=training_model[kind].predict(t)
    a.append(pred_y[0])
    b.append(k)
    #print num, i, dis_erro(pred_y, array([k])),i[training_bssid[kind]],training_bssid[kind],kind
    num +=1
a=array(a)
b=array(b)
print dis_erro(a,b)
print time.time()-start

