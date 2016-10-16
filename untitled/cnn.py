import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import cPickle as pickle



def dis_erro(pred_y,rel_y):
    diffMat = pred_y - rel_y
    sqDiff = diffMat ** 2
    sqDis = sqDiff.sum(1)
    dis_erro_values = sqDis ** 0.5
    ave_erro1 = np.mean((dis_erro_values).tolist())

    print '\n',ave_erro1
    return ave_erro1

# define model structure
def baseline_model():
    model = Sequential()
    model.add(Dense(21, input_dim=21, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=150, input_dim=21, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=300, input_dim=150,init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=450, input_dim=300, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(450,  init='normal',activation='softmax'))
    #model.summary()
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


f = open('load_data_part.pkl','r+')
data= pickle.load(f)

training_x, test_x, training_y,test_y =cross_validation.train_test_split(data[:,:-2],data[:,-2:], test_size = 0.02, random_state = 3)


Y = range(len(training_x))
print training_x.shape

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=50, batch_size=120)

#estimator = KerasClassifier(baseline_model)
estimator.fit((100-training_x)/50.0,dummy_y)

# make predictions

pred = estimator.predict((100-test_x)/50.0)

# inverse numeric variables to initial categorical labels
#init_lables = encoder.inverse_transform(pred)

dis_erro(training_y[pred,:],test_y)


#print pred,'\n','#########',test_y



