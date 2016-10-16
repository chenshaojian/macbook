# -*- coding: utf-8 -*-

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
from numpy import *
#set_printoptions(threshold='nan')


##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)
#X = StandardScaler().fit_transform(X)
f = open('load_data_local.pkl','r+')
data= pickle.load(f)
print data.shape
training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.01, random_state = 3)


##############################################################################
# Compute DBSCAN
#db = DBSCAN(eps=0.15, min_samples=50).fit(training_x)
#labels = db.labels_
def dis_erro(pred_y,rel_y):
    diffMat = pred_y - rel_y
    sqDiff = diffMat ** 2
    sqDis = sqDiff.sum(1)
    dis_erro_values = sqDis ** 0.5
    ave_erro1 = mean((dis_erro_values).tolist())
    return ave_erro1

knn=KNeighborsClassifier(n_neighbors=3,algorithm='brute').fit(training_x,training_y)
pred_y=knn.predict(test_x)
print dis_erro(pred_y,test_y)



kmeans=KMeans(n_clusters=3,n_init=10,max_iter=1000,random_state=100)
tr_labels=kmeans.fit_predict(training_x)
tt_labels=kmeans.predict(test_x)
knn_models={}
for i in set(tr_labels):
    knn_models[i]=KNeighborsClassifier(n_neighbors=1,algorithm='brute').fit(training_x[tr_labels==i], training_y[tr_labels==i])
pred_y=[]
rel_y=[]
for i in set(tt_labels):
    pred_y=pred_y+knn_models[i].predict(test_x[tt_labels==i]).tolist()
    rel_y=rel_y+test_y[tt_labels==i].tolist()
pred_y=array(pred_y)
rel_y=array(rel_y)
print dis_erro(pred_y,rel_y)




print  set(tr_labels)

# Number of clusters in tr_labels, ignoring noise if present.
n_clusters_ = len(set(tr_labels)) - (1 if -1 in tr_labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)


#############################################################################
#Plot result
import matplotlib.pyplot as plt
# Black removed and is used for noise instead.
unique_labels = set(tr_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


for k, col in zip(unique_labels, colors):
    if k==-1:
        markersize = 6
    else:
        markersize=12
    plt.plot(training_y[tr_labels==k, 0], training_y[tr_labels==k, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=markersize)


plt.plot(rel_y[:, 0], rel_y[:, 1], '*', markerfacecolor=[0.5,0.5,1],
         markeredgecolor='k', markersize=markersize)
plt.plot(pred_y[:, 0], pred_y[:, 1], '*', markerfacecolor=[1, 1, 1],
         markeredgecolor='k', markersize=markersize)
for i,j in zip(rel_y,pred_y):
    plt.plot([i[0], j[0]], [i[1], j[1]])

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
