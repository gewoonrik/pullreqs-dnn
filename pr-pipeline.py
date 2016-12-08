#!/usr/bin/env python

from __future__ import print_function
import urllib
import pandas as pd
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding, Bidirectional

np.random.seed(1337)
pd.set_option('display.max_rows', 10)

## Files
ORIG_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/data.csv"
ORIG_DATA_FILE = "pr-data.csv"
DIFFS_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/pullreq-patches.tar.gz"
DIFFS_DIR = "pullreq-patches"
DIFFS_FILE = "pullreq-patches.tar.gz"

## Tokenizer config
max_features = 20000
maxlen = 100
batch_size = 32

## Dataset splitter config
validation_split = 0.2


def create_dataset(prefix=""):
    if not os.path.exists(ORIG_DATA_FILE):
        print("Downloading pull request data file")
        urllib.urlretrieve(ORIG_DATA_URL, ORIG_DATA_FILE)

    if not os.path.exists(DIFFS_DIR):
        print("Downloading pull request diffs")
        import tarfile

        urllib.urlretrieve(DIFFS_DATA_URL, DIFFS_FILE)
        tar = tarfile.open(DIFFS_FILE, "r:gz")
        tar.extractall()
        tar.close()

    pullreqs = pd.read_csv('pr-data.csv')
    pullreqs.set_index(['project_name', 'github_id'])

    text_map = {}
    label_map = pd.DataFrame(columns=('project_name', 'github_id'))
    file_counter = 0

    for name in os.listdir(DIFFS_DIR):
        try:
            file_counter += 1
            print("%s files read" % (file_counter), end='\r')
            owner, repo, github_id = name.split('@')
            project_name = "%s/%s" % (owner, repo)
            github_id = github_id.split('.')[0]

            statinfo = os.stat(os.path.join(DIFFS_DIR, name))
            if statinfo.st_size == 0:
                # Patch is zero size
                continue

            label_map = pd.concat([label_map, pd.DataFrame([[project_name, int(github_id)]],
                                                           columns=('project_name', 'github_id'))])
            text_map[name.split('.')[0]] = os.path.join(DIFFS_DIR, name)
        except:
            pass

    print("Loaded %s patches" % len(text_map))

    label_map = pd.merge(label_map, pullreqs, how='left')[['project_name', 'github_id', 'merged']]
    label_map['name'] = label_map[['project_name', 'github_id']].apply(
        lambda x: "%s@%d" % (x[0].replace('/', '@'), x[1]),
        axis=1)
    # Balancing the dataset
    unmerged = label_map[label_map['merged'] == False]
    merged = label_map[label_map['merged'] == True].sample(n=len(unmerged))
    label_map = pd.concat([unmerged, merged]).sample(frac=1)

    print("After balancing: %s patches" % len(label_map))

    texts = []
    labels = []
    for i, row in label_map.iterrows():
        try:
            texts.append(open(text_map[row['name']]).read())
            labels.append(int(row['merged'] * 1))
        except:
            print("Failed %s" % row['name'])
            pass

    print("Cleaning up")
    del text_map, label_map, pullreqs

    print("Examining %s texts" % len(texts))
    print("Tokenizing")
    tokenizer = Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(validation_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    pickle.dump(x_train, open('x_train%s.pcl' % prefix, 'w'))
    pickle.dump(y_train, open('y_train%s.pcl' % prefix, 'w'))
    pickle.dump(x_val, open('x_val%s.pcl' % prefix, 'w'))
    pickle.dump(y_val, open('y_val%s.pcl' % prefix, 'w'))

    return x_train, y_train, x_val, y_val


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
