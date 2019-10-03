from keras.layers import Input, Dense, Lambda, Dropout, Activation
from keras.layers import LeakyReLU
from keras.models import Model,save_model,load_model, Sequential
#from keras.objectives import binary_crossentropy
#from keras.callbacks import LearningRateScheduler
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import io
#import keras.backend as K
import tensorflow as tf
from sklearn import preprocessing
import keras.optimizers
from tensorflow import set_random_seed
from numpy.random import seed
#from sklearn.model_selection import train_test_split
#import seaborn as sns
from keras import losses
#import pickle
#import requests
#import json


def prepare(filename):

    df = pd.read_csv(filename,nrows=30)
    #print(df)

    df.iloc[:,0] = pd.to_datetime(df.iloc[:, 0])
    col_name=list(df.iloc[:,1:].columns.values)
    df = np.asarray(df)
    index = df[:, 0]
    df=df[:,1:]
    preprocessed_data = np.asarray(df,dtype=np.float)
    scaler = preprocessing.MinMaxScaler()
    preprocessed_data = scaler.fit_transform(preprocessed_data)
    bol = np.all(preprocessed_data == preprocessed_data[0,:], axis = 0)
    bol = np.invert(bol)
    # print(bol.shape)
    preprocessed_data = preprocessed_data[:, bol]
    mask = np.all(np.isnan(preprocessed_data), axis=1)
    mask = np.invert(mask)
    preprocessed_data = preprocessed_data[mask]
    # print(np.sum(bol))
    index = index[mask]
    #print(index)
    return index, preprocessed_data, bol,df,col_name

def sensor_plot(sensor_name,sensor,data,predicted=None,anamoly=None):
    sensor_name=np.asarray(sensor_name)
    data=np.asarray(data)
    sensor=np.asarray(sensor,dtype=int)
    #print(data[:,sensor].shape)
    #print(data[:,sensor])
    return sensor_name[sensor], (data[:,sensor].transpose()).tolist()

#df = pd.read_csv(r"C:\Users\AYUSHI\Desktop\cleaned_data.csv",nrows=2000)
#df = pd.read_csv(r"export_dataframe.csv")
#pd.DataFrame.to_csv(df,r"C:\Users\AYUSHI\Desktop\export_dataframe.csv",index=False)
#pd.DataFrame.to_csv(df,"export_dataframe.csv",index=False)

#print(df.iloc[:, -1])
#pd.DataFrame.to_csv(df.iloc[:,:-2],r"C:\Users\AYUSHI\Desktop\export_dataframe.csv",index=False)
#pd.DataFrame.to_csv(df.iloc[:,:-2],"export_dataframe.csv",index=False)

#df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
#print(df.iloc[:, 0])


def model_create(data, m=50, n_z=6, n_epoch=10, split=0.1):
    m = int(m)
    #print(data.shape)
    n_z = int(n_z)
    n_epoch = int(n_epoch)
    split = int(split)/100
    split=int(np.round(split*data.shape[0]))
    input_shape = data.shape[1]
    i_s1 = round(input_shape * 0.75)
    i_s2 = round(input_shape * 0.5)
    i_s3 = round(input_shape * 0.25)

    #def leakyrelu(x):
    #    return K.maximum(0.3 * x, x)

    seed(10)
    set_random_seed(10)

    b = 0.1
    model = Sequential()
    # model.add(Dense(input_shape,input_shape=(input_shape,),activation=leakyrelu))
    #model.add(Dense(i_s1, input_shape=(input_shape,), activation=leakyrelu))
    model.add(Dense(i_s1, input_dim=input_shape))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(i_s2))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(i_s3))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(n_z))
    model.add(Dropout(b))
    model.add(Dense(i_s3))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(i_s2))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(i_s1))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(b))
    model.add(Dense(input_shape, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    history = model.fit(data[:split], data[:split], batch_size=m, epochs=n_epoch, verbose=1)
    model.save("model_save.h5")
    #pickle.dump(model, open('model_saved.pkl', 'wb'))
    #pred = model.predict(data, m)
    #test_pred = vae.predict(data,m)
    # return pred,test_pred,test_data
    return history.history['loss'], split

def predict_on_data(data):
    # print(data.shape)
    # print(type(data))
    # print(type(np.asarray(data)))
    # print((np.asarray(data)).shape)
    # print("her-e1")
    model = load_model('model_save.h5')
    #print("here2")
    pred = model.predict(np.asarray(data))
    #print(type(pred))
    #print(pred)
    return pred

def loss(data,pred):
    def vae_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, preferred_dtype=tf.float64)
        y_pred = tf.convert_to_tensor(y_pred, preferred_dtype=tf.float64)
        recon = losses.binary_crossentropy(y_true, y_pred)
        return recon

    loss = vae_loss(data, pred)
    with tf.Session() as s:
        global sav
        sav = np.round(s.run(loss), 2)
    return sav


def anamoly_calc(sav, data, thresh_loss, window_size):
    sav=np.asarray(sav)
    data=np.asarray(data)
    print(data.shape)
    window_size=int(window_size)
    thresh_loss=float(thresh_loss)
    # ana_index=index[sav>160]
    # ana_data=data[sav>160]
    aray = sav > thresh_loss
    hel = np.nonzero(aray)
    hel = (np.asarray(hel))
    hel = np.reshape(hel, hel.shape[1])
    #print(hel.shape)
    #print(aray.shape)
    #print(inv.shape)
    #print(hel.shape)
    # per_index=index[sav<150]
    # per_data=data[sav<150]
    # print(ana_index.shape)
    # print(ana_data.shape)
    # print(per_index.shape)
    # print(per_data.shape)

    window = window_size
    lis = []
    prev = hel[0]
    av = data[prev - window:prev]
    mean_av = np.mean(av, axis=0)
    lis.append(np.abs(mean_av - data[hel[0]]))
    diff = 0

    for i in range(1, hel.shape[0]):
        if ((prev + 1) != hel[i]):
            diff = (hel[i - 1] - hel[i])
            if (diff <= window):
                for j in range(1, diff):
                    np.delete(av, 0, axis=0)
                    av.append(data[prev + j])
            else:
                av = data[hel[i] - window:hel[i]]
            mean_av = np.mean(av, axis=0)
            lis.append(np.abs(mean_av - data[hel[i]]))

        else:
            mean_av = np.mean(av, axis=0)
            lis.append(np.abs(mean_av - data[hel[i]]))
        prev = hel[i]

    lis = np.asarray(lis)
    #print(lis.shape)
    vec = np.sum(lis, axis=1)
    # print(vec.shape)
    lis = (lis / vec[:, None]) * 100
    max= np.amin(np.amax(np.asarray(lis),axis=1))
    #print(max)
    return (lis,max, aray)


def anamoly_calc2(sav, data, thresh_loss, pred_data):
    sav=np.asarray(sav)
    data=np.asarray(data)
    pred_data=np.asarray(pred_data)
    #print(data.shape)
    thresh_loss=float(thresh_loss)
    # ana_index=index[sav>160]
    # ana_data=data[sav>160]
    aray = sav > thresh_loss
    hel = np.nonzero(aray)
    hel = (np.asarray(hel))
    hel = np.reshape(hel, hel.shape[1])

    lis=np.abs(pred_data[hel] - data[hel])
    print(lis.shape)
    print(aray.shape)
    vec = np.sum(lis, axis=1)
    # print(vec.shape)
    lis = (lis / vec[:, None]) * 100
    max= np.amin(np.amax(np.asarray(lis),axis=1))
    #print(max)
    return (lis,max, aray)


def final_calc(thresh_percen,lis2,col_name,bol):

    thresh_percen=float(thresh_percen)
    #print(type(lis2))
    col_name=np.asarray(col_name)
    #print(type(bol))
    #print(type(index))
    #print(type(aray))
    sensors=col_name[bol]
    #print("--------",thresh_percen)
    #print(type(lis2))
    #print(lis2.shape)
    gre = lis2 > thresh_percen
    #print(gre.shape)
    ana_sensors = []
    ana_per = []
    j = 0
    for i in gre:
        temp = lis2[j]
        ana_sensors.append(sensors[i])
        ana_per.append(temp[i])
        j = j + 1
    #ana_index = index[aray]
    #print(ana_index)
    #print(ana_index.shape)
    #print(ana_sensors)
    #print(ana_per)
    return ana_sensors,ana_per