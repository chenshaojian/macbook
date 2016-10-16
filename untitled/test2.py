#-*-coding:utf-8 -*-
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle
from numpy import *
import time
#set_printoptions(threshold='nan')

def dis_erro(pred_y,rel_y):
    diffMat = pred_y - rel_y
    sqDiff = diffMat ** 2
    sqDis = sqDiff.sum(1)
    dis_erro_values = sqDis ** 0.5
    ave_erro1 = mean((dis_erro_values).tolist())
    return ave_erro1
##############################################################################
# Generate sample data
start=time.time()
f = open('load_data_part.pkl','r+')
data= pickle.load(f)
training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.05, random_state = 3)
print '训练数据量:',len(training_x),' 测试数据量:',len(test_x)

knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute',weights = 'distance').fit(training_x, training_y)
pred_y=knn.predict(test_x)

a=[]
b=[]
num=0

for i,k in zip(test_x,test_y):
    t=i[i>0]
    training1_x = []
    for j in training_x[:,i>0]:
        if any(i):
            training1_x.append(j)
    training1_x=array(training1_x)
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='brute',weights='distance').fit(training1_x, training_y)
    pred_y=knn.predict([t])
    dis, index = knn.kneighbors([t])
    print '##########'
    print k
    print dis
    print training_y[index]
    a.append(pred_y[0])
    b.append(k)
    #print num,i, dis_erro(pred_y,array([k]))
    num +=1
a=array(a)
b=array(b)
print dis_erro(a,b)
print time.time()-start