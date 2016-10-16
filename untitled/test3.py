from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
from numpy import *
from pandas import Series,DataFrame
import time
import pandas as pd
#set_printoptions(threshold='nan')
def dis_erro(pred_y,rel_y):
    diffMat = pred_y - rel_y
    sqDiff = diffMat ** 2
    sqDis = sqDiff.sum(1)
    dis_erro_values = sqDis ** 0.5
    ave_erro1 = mean((dis_erro_values).tolist())
    return ave_erro1
##############################################################################
start=time.time()
f = open('load_data2.pkl','r+')
data= pickle.load(f)

training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.1, random_state = 3)
training_model={}
training_bssid={}

(row,col)=data.shape
for i in range(row):
    for j in range(col):
        if data[i][j]!=100:
            data[i][j]=int((100-data[i][j])/10)
        else:
            data[i][j]=0
data=DataFrame(data[:,:-2])
#print data[(data[0]>0 )&(data[1]>90)]
cf=data.corr(method='pearson')
#print  d1.sort_index(by=0,ascending=False)[0]
#bssid = cf.sort_index(by=0, ascending=False)
#bssid=cf[cf[0]>0][0]

for i in range(21):
      bssid=cf[cf[i]>0.3][i]
      #print len(bssid)
      #print cf[i]
      print list(bssid.keys())
      training_bssid[i]=list(bssid.keys())
      training_model[i]=KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(training_x[:,training_bssid[i]], training_y)
a=[]
b=[]
num=0
for i,k in zip(test_x,test_y):
    minnum=100
    c=list(i)
    for r in i:
        if r<minnum and r>0:
            minnum=r
    kind=c.index(minnum)
    t=i[training_bssid[kind]]
    pred_y=training_model[kind].predict([t])
    a.append(pred_y[0])
    b.append(k)

    #print num, i, dis_erro(pred_y, array([k])),i[training_bssid[kind]],training_bssid[kind],kind
    num +=1
a=array(a)
b=array(b)
print dis_erro(a,b)
print time.time()-start
