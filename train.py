#!/usr/bin/env python

from __future__ import print_function

import os
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding, Bidirectional

from config import *
from preprocess import create_dataset

np.random.seed(1337)

if not os.path.exists('x_train.pcl') or \
        not os.path.exists('y_train.pcl') or \
        not os.path.exists('x_val.pcl') or \
        not os.path.exists('y_val.pcl'):
    print("Creating data set")

    x_train, y_train, x_val, y_val = create_dataset()

else:
    print("Loading existing data set")
    x_train = pickle.load(open('x_train.pcl'))
    y_train = pickle.load(open('y_train.pcl'))
    x_val = pickle.load(open('x_val.pcl'))
    y_val = pickle.load(open('y_val.pcl'))

print("Training on %d merged, %d unmerged PRs" % (y_train[y_train == 1].size, y_train[y_train == 0].size))

model = Sequential()
model.add(Embedding(max_features, 512, dropout=0.2))
# model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Bidirectional(LSTM(256, dropout_W=0.2, dropout_U=0.2)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'fmeasure'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=30,
          validation_data=(x_val, y_val))
score, acc = model.evaluate(x_val, y_val, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
