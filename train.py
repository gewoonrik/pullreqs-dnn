#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>
#


import argparse
import pickle
import json

from keras.models import Model
from keras.layers import Input, Dense, merge, LSTM, Embedding
from keras.layers import LSTM, Dense, Activation, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lstm_diff_output', type=int, default=256)
parser.add_argument('--lstm_title_output', type=int, default=256)
parser.add_argument('--lstm_comment_output', type=int, default=256)
parser.add_argument('--diff_embedding_output', type=int, default=512)
parser.add_argument('--title_embedding_output', type=int, default=512)
parser.add_argument('--comment_embedding_output', type=int, default=512)
parser.add_argument('--checkpoint', type=bool, default=False)
parser.add_argument('--max_diff_sequence_length', type=int, default=100)
parser.add_argument('--max_title_sequence_length', type=int, default=100)
parser.add_argument('--max_comment_sequence_length', type=int, default=100)

args = parser.parse_args()

print("Loading data set for prefix %s" % args.prefix)
diff_train = pickle.load(open(diff_train_file % args.prefix))
title_train = pickle.load(open(title_train_file % args.prefix))
comment_train = pickle.load(open(comment_train_file % args.prefix))
y_train = pickle.load(open(y_train_file % args.prefix))

diff_val = pickle.load(open(diff_val_file % args.prefix))
title_val = pickle.load(open(title_val_file % args.prefix))
comment_val = pickle.load(open(comment_val_file % args.prefix))
y_val = pickle.load(open(y_val_file % args.prefix))

diff_test = pickle.load(open(diff_test_file % args.prefix))
title_test = pickle.load(open(title_test_file % args.prefix))
comment_test = pickle.load(open(comment_test_file % args.prefix))
y_test = pickle.load(open(y_test_file % args.prefix))

config = pickle.load(open(config_file % args.prefix))

print("Training on %d merged, %d unmerged PRs" % (y_train[y_train == 1].size,
                                                  y_train[y_train == 0].size))
config.update(vars(args))
print("Training configuration:")
print json.dumps(config, indent=1)


diff_input = Input(shape=(config['max_diff_length'],), dtype='int32', name='diff_input')
diff_embedding = Embedding(config['diff_vocabulary_size'], args.diff_embedding_output, dropout=args.dropout)(diff_input)
diff_lstm = LSTM(args.lstm_diff_output, consume_less='gpu', dropout_W=args.dropout, dropout_U=args.dropout)(diff_embedding)
diff_auxiliary_output = Dense(1, activation='sigmoid', name='diff_aux_output')(diff_lstm)


comment_input = Input(shape=(config['max_comment_length'],), dtype='int32', name='comment_input')
comment_embedding = Embedding(config['comment_vocabulary_size'], args.comment_embedding_output, dropout=args.dropout)(comment_input)
comment_lstm = LSTM(args.lstm_comment_output, consume_less='gpu', dropout_W=args.dropout, dropout_U=args.dropout)(comment_embedding)
comment_auxiliary_output = Dense(1, activation='sigmoid', name='comment_aux_output')(comment_lstm)

title_input = Input(shape=(config['max_title_length'],), dtype='int32', name='title_input')
title_embedding = Embedding(config['title_vocabulary_size'], args.comment_embedding_output, dropout=args.dropout)(title_input)
title_lstm = LSTM(args.lstm_title_output, consume_less='gpu', dropout_W=args.dropout, dropout_U=args.dropout)(title_embedding)
title_auxiliary_output = Dense(1, activation='sigmoid', name='title_aux_output')(title_lstm)


merged = merge([diff_lstm, comment_lstm, title_lstm], mode='concat')

dense = Dense(128, activation='relu')(merged)
dense = Dense(128, activation='relu')(dense)
dense = Dense(128, activation='relu')(dense)

main_output = Dense(1, activation='sigmoid', name='main_output')(dense)

model = Model(input=[diff_input, comment_input, title_input], output=[main_output, diff_auxiliary_output, comment_auxiliary_output, title_auxiliary_output])


optimizer = RMSprop(lr = 0.005)

model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', 'fmeasure'],
            loss_weights=[1., 0.3, 0.1, 0.1])

print('Train...')
csv_logger = CSVLogger('traininglog_%s.csv' % args.prefix)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

callbacks = [csv_logger, early_stopping, reduce_lr]

if args.checkpoint:
    checkpoint = ModelCheckpoint(checkpoint_file % args.prefix, monitor='val_loss')
    callbacks.append(checkpoint)

model.fit([diff_train, comment_train, title_train], [y_train, y_train, y_train, y_train], batch_size=args.batch_size, nb_epoch=args.epochs,
          validation_data=([diff_val, comment_val, title_val], [y_val, y_val, y_val, y_val]), callbacks=callbacks)

[(main_score, main_acc), (diff_score, diff_acc), (comment_score, comment_acc), (title_score,title_acc)] = model.evaluate([diff_test, comment_test, title_test], [y_test,y_test,y_test, y_test], batch_size=args.batch_size)
print('Test main score:', main_score)
print('Test main accuracy:', main_acc)
print('Test diff score:', diff_score)
print('Test diff accuracy:', diff_acc)
print('Test comment score:', comment_score)
print('Test comment accuracy:', comment_acc)
print('Test title score:', title_score)
print('Test title accuracy:', title_acc)