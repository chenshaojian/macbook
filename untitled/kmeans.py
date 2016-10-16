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


f = open('load_data_local.pkl','r+')
data=pickle.load(f)
training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.001, random_state = 3)
kmeans=KMeans(n_clusters=3,n_init=100,max_iter=1000,random_state=100)
tr_labels=kmeans.fit_predict(training_x)
print data.shape



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
    plt.plot(training_y[tr_labels==k, 0]/100, training_y[tr_labels==k, 1]/100, 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=markersize)
plt.show()


