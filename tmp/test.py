import os
import matplotlib.image as img
import numpy as np
from PIL import Image
import pandas as pd
import ast 
import keras.models
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from numpy import array  

def onehot(dic,allg):
    dic = ast.literal_eval(dic)
    out = [0] * len(allg)
    tmp = 0
    outy = [0] * len(allg)
    for g in dic:
        try:
            out[allg.index(g['name'])] =1
            if tmp == 0:
                outy[allg.index(g['name'])] =1
                tmp = 1
        except:
            continue
    outy = array(outy)
    return out,outy


def split_data(x,y, proportion):
    n = len(x)
    tmp = 0
    x_test = [] 
    y_test = []
    x_train = [] 
    y_train = []
    for xx, yy in zip(x,y):
        if n* proportion > tmp:
            x_test.append(xx)
            y_test.append(yy)
        else:
            x_train.append(xx)
            y_train.append(yy)
        tmp+= 1
    x_test = array(x_test)
    y_test = array(y_test)
    x_train = array(x_train)
    y_train = array(y_train)
    return x_test, y_test , x_train, y_train

file_dir = "../data/movies_metadata.csv"
data = pd.read_csv(file_dir)
data = data['genres']

image_dir = "../images"
files = os.listdir("../images")

genre_all = []
allg = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western','TV Movie']
i = 0
model=keras.models.load_model('cnn_model')

x = np.empty((0,0,0))
y = np.empty((0,0))
t = 1
for image in files:    
    if i < 70:
        i += 1
        add = Image.open(image_dir+"/"+str(image))
        add = add.resize((278,185))
        add = np.array(add)
        x = np.append(x,add)
        
        all_add, y_add = onehot(data.iloc[int(image[:-4])],allg) 
        y =np.append(y, y_add)
        # genre_all.append(all_add)
    else:
        try:
            t+=1 
            x = x.reshape((i,278,185,3))
            y = y.reshape((i,20))
            x_test, y_test , x_train, y_train = split_data(x,y,0.1)
            print(y_train)
            predict= model.predict(x_train)
            # print(predict)
            i = 0
            x = np.empty((0,0,0))
            y = np.empty((0,0))
        except:
            continue

model.save("cnn_model")



# batch_size = 32

# hist = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(X_train) / 32,
#       epochs=epochs,
#     validation_data=test_generator,
#   #   callbacks=[callback],
#     validation_steps = len(X_test)/32)
