#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>, Rik Nijessen <riknijessen@gmail.com>
#


from __future__ import print_function

import os
import pickle
import urllib
import numpy as np
import argparse

from config import *
from code_tokenizer import CodeTokenizer
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

def read_title_and_comments(file):
    str = open(file).read()
    splitted = str.split("\n")
    title = splitted[0]
    # remove title and empty space
    comment = str[2:]
    return title, comment

@timeit
def tokenize_code(texts, vocabulary_size, maxlen):
    print("Tokenizing")
    tokenizer = CodeTokenizer(nb_words=vocabulary_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique code tokens.' % len(word_index))

    return pad_sequences(sequences, maxlen=maxlen)

@timeit
def tokenize_text(texts, vocabulary_size, maxlen):
    print("Tokenizing")
    tokenizer = Tokenizer(nb_words=vocabulary_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique text tokens.' % len(word_index))

    return pad_sequences(sequences, maxlen=maxlen)

@timeit
def create_dataset(prefix="default", balance_ratio=1, num_diffs=-1,
                   langs=[], validation_split=0.2,
                   diff_vocabulary_size=20000,
                   comment_vocabulary_size=20000,
                   title_vocabulary_size=20000,

                   max_diff_length=100,
                   max_comment_length=100,
                   max_title_length=100):
    """
    Create a dataset for further processing
    :param prefix: Name for the dataset
    :param balance_ratio: The ratio between merged and unmerged PRs to include
    :param num_diffs: Total number of diffs to load. Any value below 1 means load all diffs.
    :param langs: Only include PRs for repos whose primary language is within this array
    :param diff_vocabulary_size: (Max) size of the diff vocabulary to use for tokenizing
    :param comment_vocabulary_size: (Max) size of the comment vocabulary to use for tokenizing
    :param title_vocabulary_size: (Max) size of the title vocabulary to use for tokenizing
    :param max_diff_length: Maximum length of the input diff sequences
    :param max_comment_length: Maximum length of the input comment sequences
    :param max_title_length: Maximum length of the input title sequences
    :return: A training and testing dataset, along with the config used to produce it
    """
    config = locals()

    pullreqs = filter_langs(load_pr_csv(), langs)
    ensure_diffs()
    #todo ensure comments and title?

    text_map = {}
    label_map = pd.DataFrame(columns=('project_name', 'github_id'))
    files_read = files_examined =0
    project_names = set(pd.Series.unique(pullreqs['project_name']))

    for name in os.listdir(DIFFS_DIR):
        files_examined += 1
        if num_diffs > 0 and files_read >= num_diffs:
            break

        try:
            owner, repo, github_id = name.split('@')
            project_name = "%s/%s" % (owner, repo)

            if project_name not in project_names:
                continue

            github_id = github_id.split('.')[0]

            statinfo = os.stat(os.path.join(DIFFS_DIR, name))
            if statinfo.st_size == 0:
                # Diff is zero size
                continue

            label_map = pd.concat([label_map, pd.DataFrame([[project_name, int(github_id)]],
                                                           columns=('project_name', 'github_id'))])
            text_map[name.split('.')[0]] = name
            files_read += 1
        except:
            pass

        print("%s diffs examined, %s diffs matching" % (files_examined, files_read) , end='\r')

    print("\nLoaded %s diffs" % len(text_map))

    label_map = pd.merge(label_map, pullreqs, how='left')[['project_name', 'github_id', 'merged']]
    label_map['name'] = label_map[['project_name', 'github_id']].apply(
        lambda x: "%s@%d" % (x[0].replace('/', '@'), x[1]),
        axis=1)

    # Balancing the dataset
    label_map = balance(label_map, balance_ratio)
    print("After balancing: %s diffs" % len(label_map))

    diffs = []
    titles = []
    comments = []
    labels = []
    successful = failed = 0
    for i, row in label_map.iterrows():
        try:
            diff_file = os.path.join(DIFFS_DIR, text_map[row['name']])
            comment_file = os.path.join(TXTS_DIR, text_map[row['name']].replace(".patch",".txt"))

            diff = open(diff_file).read()
            title, comment = read_title_and_comments(comment_file)

            diffs.append(diff)
            titles.append(title)
            comments.append(comment)
            labels.append(int(row['merged'] * 1))
            successful += 1
        except:
            failed += 1
            pass
        print("%s diffs loaded, %s diffs failed" % (successful, failed), end='\r')

    print("")
    diff_tokens = tokenize_code(diffs, diff_vocabulary_size, max_diff_length)
    comment_tokens = tokenize_text(comments, comment_vocabulary_size, max_comment_length)
    title_tokens = tokenize_text(titles, title_vocabulary_size, max_title_length)

    labels = np.asarray(labels)
    print('Shape of diff tensor:', diff_tokens.shape)
    print('Shape of comment tensor:', comment_tokens.shape)
    print('Shape of title tensor:', title_tokens.shape)

    print('Shape of label tensor:', labels.shape)

    # Random selection split between training and testing
    indices = np.arange(diff_tokens.shape[0])
    np.random.shuffle(indices)
    data_diff = diff_tokens[indices]
    data_comment = comment_tokens[indices]
    data_title = title_tokens[indices]

    labels = labels[indices]
    nb_validation_samples = int(validation_split * data_diff.shape[0])

    diff_train = data_diff[:-nb_validation_samples]
    comment_train = data_comment[:-nb_validation_samples]
    title_train = data_title[:-nb_validation_samples]

    y_train = labels[:-nb_validation_samples]
    diff_val = data_diff[-nb_validation_samples:]
    comment_val = data_comment[-nb_validation_samples:]
    title_val = data_title[-nb_validation_samples:]

    y_val = labels[-nb_validation_samples:]

    # Save dataset
    with open(diff_train_file % prefix, 'w') as f:
        pickle.dump(diff_train, f)

    with open(comment_train_file % prefix, 'w') as f:
        pickle.dump(comment_train, f)

    with open(title_train_file % prefix, 'w') as f:
        pickle.dump(title_train, f)

    with open(y_train_file % prefix, 'w') as f:
        pickle.dump(y_train, f)

    with open(diff_val_file % prefix, 'w') as f:
        pickle.dump(diff_val, f)

    with open(comment_val_file % prefix, 'w') as f:
        pickle.dump(comment_val, f)

    with open(title_val_file % prefix, 'w') as f:
        pickle.dump(title_val, f)


    with open(y_val_file % prefix, 'w') as f:
        pickle.dump(y_val, f)

    with open(config_file % prefix, 'w') as f:
        pickle.dump(config, f)

    return diff_train, comment_train, title_train, y_train, diff_val, comment_val, title_val, y_val, config


np.random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--balance_ratio', type=float, default=1)
parser.add_argument('--num_diffs', type=int, default=-1)
parser.add_argument('--langs', nargs="*", default='')
parser.add_argument('--validation_split', type=float, default=0.2)
parser.add_argument('--diff_vocabulary_size', type=int, default=20000)
parser.add_argument('--comment_vocabulary_size', type=int, default=20000)
parser.add_argument('--title_vocabulary_size', type=int, default=20000)
parser.add_argument('--max_diff_sequence_length', type=int, default=100)
parser.add_argument('--max_comment_sequence_length', type=int, default=100)
parser.add_argument('--max_title_sequence_length', type=int, default=100)

args = parser.parse_args()

if __name__ == '__main__':
    create_dataset(args.prefix, args.balance_ratio, args.num_diffs, args.langs,
                   args.validation_split, args.diff_vocabulary_size, args.comment_vocabulary_size, args.title_vocabulary_size, args.max_diff_sequence_length, args.max_comment_sequence_length, args.max_title_sequence_length)