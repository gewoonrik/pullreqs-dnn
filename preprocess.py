#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>
#


from __future__ import print_function

import os
import pickle
import urllib
import numpy as np
import argparse

from config import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


@timeit
def load_pr_csv():
    """
    Load (download if needed) the original PR dataset, including all engineered features
    :return: A pandas dataframe with all data loaded
    """
    if not os.path.exists(ORIG_DATA_FILE):
        print("Downloading pull request data file")
        urllib.urlretrieve(ORIG_DATA_URL, ORIG_DATA_FILE)

    print("Loading pull requests file")
    pullreqs = pd.read_csv(ORIG_DATA_FILE)
    pullreqs.set_index(['project_name', 'github_id'])
    return pullreqs


def ensure_diffs():
    """
    Make sure that the PR diffs have been downloaded in the appropriate dir
    """
    if not os.path.exists(DIFFS_DIR):
        print("Downloading pull request diffs")
        import tarfile

        urllib.urlretrieve(DIFFS_DATA_URL, DIFFS_FILE)
        tar = tarfile.open(DIFFS_FILE, "r:gz")
        tar.extractall()
        tar.close()


def filter_langs(pullreqs, langs):
    """
    Apply a language filter on the pullreqs dataframe
    """
    if len(langs) > 0:
        print("Filtering out pull requests not in %s" % langs)
        pullreqs = pullreqs[pullreqs['lang'].str.lower().isin([x.lower() for x in langs])]

    return pullreqs


def balance(pullreqs, balance_ratio):
    """
    Balance the dataset between merged and unmerged pull requests
    """
    unmerged = pullreqs[pullreqs['merged'] == False]
    if len(unmerged) == 0:
        raise Exception("No unmerged pull requests in filtered dataset")

    merged = pullreqs[pullreqs['merged'] == True].sample(n=(len(unmerged) * balance_ratio))

    return pd.concat([unmerged, merged]).sample(frac=1)


@timeit
def tokenize(texts, vocabulary_size, maxlen):
    print("Tokenizing")
    tokenizer = Tokenizer(nb_words=vocabulary_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    return pad_sequences(sequences, maxlen=maxlen)


@timeit
def create_dataset(prefix="default", balance_ratio=1, num_diffs=-1,
                   langs=[], validation_split=0.2, vocabulary_size=20000,
                   maxlen=100):
    """
    Create a dataset for further processing
    :param prefix: Name for the dataset
    :param balance_ratio: The ratio between merged and unmerged PRs to include
    :param num_diffs: Total number of diffs to load. Any value below 1 means load all diffs.
    :param langs: Only include PRs for repos whose primary language is within this array
    :param vocabulary_size: (Max) size of the vocabulary to use for tokenizing
    :param maxlen: Maximum length of the input sequences
    :return: A training and testing dataset, along with the config used to produce it
    """
    config = locals()

    pullreqs = filter_langs(load_pr_csv(), langs)
    ensure_diffs()

    text_map = {}
    label_map = pd.DataFrame(columns=('project_name', 'github_id'))
    file_counter = 0

    for name in os.listdir(DIFFS_DIR):
        if num_diffs > 0 and file_counter >= num_diffs:
            break

        try:
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
            file_counter += 1
        except:
            pass

        print("%s diffs read" % file_counter, end='\r')

    print("Loaded %s diffs" % len(text_map))

    label_map = pd.merge(label_map, pullreqs, how='left')[['project_name', 'github_id', 'merged']]
    label_map['name'] = label_map[['project_name', 'github_id']].apply(
        lambda x: "%s@%d" % (x[0].replace('/', '@'), x[1]),
        axis=1)

    # Balancing the dataset
    label_map = balance(label_map, balance_ratio)
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

    tokens = tokenize(texts, vocabulary_size, maxlen)
    labels = np.asarray(labels)
    print('Shape of data tensor:', tokens.shape)
    print('Shape of label tensor:', labels.shape)

    # Random selection split between training and testing
    indices = np.arange(tokens.shape[0])
    np.random.shuffle(indices)
    data = tokens[indices]
    labels = labels[indices]
    nb_validation_samples = int(validation_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    # Save dataset
    with open(x_train_file % prefix, 'w') as f:
        pickle.dump(x_train, f)

    with open(y_train_file % prefix, 'w') as f:
        pickle.dump(y_train, f)

    with open(x_val_file % prefix, 'w') as f:
        pickle.dump(x_val, f)

    with open(y_val_file % prefix, 'w') as f:
        pickle.dump(y_val, f)

    with open(config_file % prefix, 'w') as f:
        pickle.dump(config, f)

    return x_train, y_train, x_val, y_val, config


np.random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--balance_ratio', type=float, default=1)
parser.add_argument('--num_diffs', type=int, default=-1)
parser.add_argument('--langs', nargs="*", default='')
parser.add_argument('--validation_split', type=float, default=0.2)
parser.add_argument('--vocabulary_size', type=int, default=20000)
parser.add_argument('--max_sequence_length', type=int, default=100)

args = parser.parse_args()

if __name__ == '__main__':
    create_dataset(args.prefix, args.balance_ratio, args.num_diffs, args.langs,
                   args.validation_split, args.vocabulary_size, args.max_sequence_length)
