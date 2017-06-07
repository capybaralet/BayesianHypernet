#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  



batch_size = 32
nb_classes = 10
#nb_epoch = 200
nb_epoch = 3
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Original size of the cifar10 dataset')
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

print ('Using poritions of the training and test sets')

X_train = X_train[0:1000, 0:3,0:32,0:32]
y_train = y_train[0:1000, 0]

X_test = X_test[0:500,0:3,0:32,0:32]
y_test = y_test[0:500,0]


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# defining the model 
model = Sequential()


# 3 layers of the CNN - including the final fully connected NN layer
# Dropout - applied after the MaxPooling stage in the CNN layers
# and after the activation stage in the final NN layer after Dense() is applied

# applying a 3x3 convolution with 32 output filters on a 32x32 image (img_rows, img_cols = 32, 32)
# applying the filter in the convolutional layer

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))   #using relu activation function
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('Shape after this /= 255')
print('X_train shape:', X_train.shape)
print('X_test shape:' , X_test.shape)


if not data_augmentation:
    print('Not using data augmentation or normalization')

    # fit - trains for the model for the given number of epochs
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)


    # score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    score1 = model.predict(X_test, batch_size=batch_size, verbose=1)
    score2 = model.predict(X_test, batch_size=batch_size, verbose=1)

    # for i in range(2):
    #     score = model.predict(X_test, batch_size=batch_size, verbose=1)
    #     print('Test score:', score)















# else:
#     print('Using real time data augmentation')

#     # this will do preprocessing and realtime data augmentation
#     datagen = ImageDataGenerator(
#         featurewise_center=True,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=True,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  #apply ZCA whitening
#         rotation_range=20,  #randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.2,  #randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.2,  #randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  #randomly flip images
#         vertical_flip=False)  #randomly flip images

#     # compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied)
#     datagen.fit(X_train)

#     for e in range(nb_epoch):
#         print('-'*40)
#         print('Epoch', e)
#         print('-'*40)
#         print('Training...')
#         # batch train with realtime data augmentation
#         progbar = generic_utils.Progbar(X_train.shape[0])
#         for X_batch, Y_batch in datagen.flow(X_train, Y_train):
#             loss = model.train_on_batch(X_batch, Y_batch)
#             progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])

#         print('Testing...')
#         # test time!
#         progbar = generic_utils.Progbar(X_test.shape[0])
#         for X_batch, Y_batch in datagen.flow(X_test, Y_test):
#             score = model.test_on_batch(X_batch, Y_batch)
#             progbar.add(X_batch.shape[0], values=[('test loss', score[0])])
