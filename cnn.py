
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import cPickle as pickle
import sys
import time
from Metrics import f1,recall,precision
from matplotlib import pyplot as plt
import keras
import pandas as pd

def convert(data, genres):
    out = [0]* len(genres)
    for i in enumerate(data):
        out[genres.index(data[i])] = 1
        break
    return out

def load_csv(file_dir,genres):
    data = pd.read_csv(file_dir)
    x = data['image']
    y = data['genre']
    y = y.apply(convert, args=(genres,))
    return x,y

file_dir = ""
x,y = load_csv(file_dir)
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=100,stratify=y,test_size=.15)
train_datagen = ImageDataGenerator(
	        rescale=1./255,
	        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train,y_train)
test_generator = test_datagen.flow(X_test,y_test)

# dimensions of images.
img_height,img_width = X_train.shape[1],X_train.shape[2]

# number of class
num_class = y_train.shape[1]

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)
print 'input shape: ',input_shape
# layer 1
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 2
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 3
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 4
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# layer 5
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 6
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 7
model.add(Dense(num_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',recall,precision])

batch_size = 32

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train) / 32,
      epochs=epochs,
    validation_data=test_generator,
  #   callbacks=[callback],
    validation_steps = len(X_test)/32)
model.save(save)
