#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>, Rik Nijessen <riknijessen@gmail.com>
#


import argparse
import pickle
import json

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lstm_output', type=int, default=300)
parser.add_argument('--embedding_output', type=int, default=500)
parser.add_argument('--checkpoint', type=bool, default=False)

args = parser.parse_args()

print("Loading data set for prefix %s" % args.prefix)
x_train = pickle.load(open(diff_train_file % args.prefix))
y_train = pickle.load(open(y_train_file % args.prefix))
x_val = pickle.load(open(diff_val_file % args.prefix))
y_val = pickle.load(open(y_val_file % args.prefix))
x_test = pickle.load(open(diff_test_file % args.prefix))
y_test = pickle.load(open(y_test_file % args.prefix))
config = pickle.load(open(config_file % args.prefix))

print("Training on %d merged, %d unmerged PRs" % (y_train[y_train == 1].size,
                                                  y_train[y_train == 0].size))
config.update(vars(args))
print("Training configuration:")
print json.dumps(config, indent=1)

model = Sequential()
model.add(Embedding(config['diff_vocabulary_size'], args.embedding_output, dropout=args.dropout))
model.add(LSTM(args.lstm_output, consume_less='gpu', dropout_W=args.dropout, dropout_U=args.dropout))
model.add(Dense(1, activation='sigmoid'))

optimizer = RMSprop(lr=0.006)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', 'fmeasure'])

print('Train...')
csv_logger = CSVLogger('traininglog_%s_model1.csv' % args.prefix)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

callbacks = [csv_logger, early_stopping, reduce_lr]

if args.checkpoint:
    checkpoint = ModelCheckpoint(checkpoint_file_1 % args.prefix, monitor='val_loss')
    callbacks.append(checkpoint)

model.fit(x_train, y_train, batch_size=args.batch_size, nb_epoch=args.epochs,
          validation_data=(x_val, y_val), callbacks=callbacks)

results = model.evaluate(x_test, y_test, batch_size=args.batch_size)
print('Test results: ', results)
print('On metrics: ', model.metrics_names)
