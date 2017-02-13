#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>, Rik Nijessen <riknijessen@gmail.com>
#


from __future__ import print_function

import pickle
import random
import urllib
import numpy as np
import argparse

from config import *
from code_tokenizer import CodeTokenizer
from my_tokenizer import MyTokenizer
from keras.preprocessing.sequence import pad_sequences


@timeit
def load_pr_csv(file):
    """
    Load a PR dataset, including all engineered features
    :return: A pandas dataframe with all data loaded
    """
    print("Loading pull requests file ", file)
    pullreqs = pd.read_csv(file)
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


def read_title_and_comments(file):
    str = open(file).read()
    splitted = str.split("\n")
    title = splitted[0]
    # remove title and empty space
    comment = str[2:]
    return title, comment

@timeit
def create_code_tokenizer(code, vocabulary_size):
    tokenizer = CodeTokenizer(nb_words=vocabulary_size)
    tokenizer.fit_on_texts(code)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer

def create_text_tokenizer(texts, vocabulary_size):
    tokenizer = MyTokenizer(nb_words=vocabulary_size)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return tokenizer


@timeit
def tokenize(tokenizer, texts, maxlen):
    print("Tokenizing")
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=maxlen)


def load_data(pullreqs):
    diffs = []
    titles = []
    comments = []
    labels = []
    successful = failed = 0
    for i, row in pullreqs.iterrows():
        try:
            name = (row['project_name']).replace('/','@')+"@"+str(row['github_id'])+'.patch'

            diff_file = os.path.join(DIFFS_DIR, name)
            comment_file = os.path.join(TXTS_DIR, name.replace(".patch",".txt"))

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
    return diffs, comments, titles, labels


@timeit
def create_dataset(prefix="default",
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

    pullreqs_train = load_pr_csv(train_csv_file % prefix)
    pullreqs_test = load_pr_csv(test_csv_file % prefix)
    pullreqs_validation = load_pr_csv(validation_csv_file % prefix)

    ensure_diffs()

    tr_diffs, tr_comments, tr_titles, tr_labels = load_data(pullreqs_train)
    val_diffs, val_comments, val_titles, val_labels = load_data(pullreqs_validation)
    te_diffs, te_comments, te_titles, te_labels = load_data(pullreqs_test)

    code_tokenizer = create_code_tokenizer(tr_diffs+val_diffs, diff_vocabulary_size)

    diff_train = tokenize(code_tokenizer, tr_diffs, max_diff_length)
    diff_val = tokenize(code_tokenizer, val_diffs, max_diff_length)
    diff_test = tokenize(code_tokenizer, te_diffs, max_diff_length)

    comment_tokenizer = create_text_tokenizer(tr_comments+val_comments, comment_vocabulary_size)

    comment_train = tokenize(comment_tokenizer, tr_comments, max_comment_length)
    comment_val = tokenize(code_tokenizer, val_comments, max_comment_length)
    comment_test = tokenize(comment_tokenizer, te_comments, max_comment_length)

    title_tokenizer = create_text_tokenizer(tr_titles+val_titles, title_vocabulary_size)

    title_train = tokenize(title_tokenizer, tr_titles, max_title_length)
    title_val = tokenize(code_tokenizer, val_titles, max_title_length)
    title_test = tokenize(title_tokenizer, te_titles, max_title_length)


    y_train = np.asarray(tr_labels)
    y_val = np.asarray(val_labels)
    y_test = np.asarray(te_labels)


    print('Shape of diff tensor:', diff_train.shape)
    print('Shape of comment tensor:', comment_train.shape)
    print('Shape of title tensor:', title_train.shape)
    print('Shape of label tensor:', y_train.shape)




    # tokenize testset
    test_tokens = tokenize(tokenizer, test_texts, maxlen)
    test_labels = np.asarray(test_labels)

    print('Shape of test_data tensor:', test_tokens.shape)
    print('Shape of test_label tensor:', test_labels.shape)

    # Save dataset
    with open(diff_vocab_file % prefix, 'w') as f:
        pickle.dump(code_tokenizer, f)

    with open(comment_vocab_file % prefix, 'w') as f:
        pickle.dump(comment_tokenizer, f)

    with open(title_vocab_file % prefix, 'w') as f:
        pickle.dump(title_tokenizer, f)

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

    # save testdata
    with open(diff_test_file % prefix, 'w') as f:
        pickle.dump(diff_test, f)

    with open(comment_test_file % prefix, 'w') as f:
        pickle.dump(comment_test, f)


    with open(title_test_file % prefix, 'w') as f:
        pickle.dump(title_test, f)

    with open(y_test_file % prefix, 'w') as f:
        pickle.dump(y_test, f)


    with open(config_file % prefix, 'w') as f:
        pickle.dump(config, f)

    return diff_train, comment_train, title_train, y_train, diff_val, comment_val, title_val, y_val, diff_test, comment_test, title_test, y_test, config

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--diff_vocabulary_size', type=int, default=50000)
parser.add_argument('--comment_vocabulary_size', type=int, default=50000)
parser.add_argument('--title_vocabulary_size', type=int, default=10000)
parser.add_argument('--max_diff_sequence_length', type=int, default=150)
parser.add_argument('--max_comment_sequence_length', type=int, default=150)
parser.add_argument('--max_title_sequence_length', type=int, default=150)


args = parser.parse_args()

if __name__ == '__main__':
    create_dataset(args.prefix, args.diff_vocabulary_size, args.comment_vocabulary_size, args.title_vocabulary_size, args.max_diff_sequence_length, args.max_comment_sequence_length, args.max_title_sequence_length)

