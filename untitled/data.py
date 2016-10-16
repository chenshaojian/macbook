from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
from sklearn.cluster import KMeans
import copy
import numpy
import os
from sklearn.cluster import DBSCAN
fname='load_data_part.pkl'
if os.path.isfile(fname):
    numpy.set_printoptions(threshold='nan')
    f = open('load_data_part.pkl', 'r+')
    data = pickle.load(f)
    print data.shape
    import pandas as pd
    df = pd.DataFrame(data)
    print df
else:
    numpy.set_printoptions(threshold='nan')
    f = open('load_data2.pkl','r+')
    data= pickle.load(f)
    data=abs(data)
    orgin_data=copy.copy(data)
    labels=data[:,-2:]
    data=data[:,:-2]
    rows,cols=data.shape
    #print data[:,[5, 6, 7, 8, 9, 10, 11, 12, 15, 16]]
    for i in range(rows):
        for j in range(cols):
            if data[i][j]==0:
                data[i][j]= 100
    clf=KMeans(n_clusters=8,random_state=50,n_init=200,max_iter=500)
    pred_k=clf.fit_predict(data)
    data= data[pred_k==7]
    labels=labels[pred_k==7]

    data_part=orgin_data[pred_k==7]
    f = open('load_data_part.pkl','wb')
    pickle.dump(data_part,f)
    f.close()


