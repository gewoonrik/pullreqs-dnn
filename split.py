
#!/usr/bin/env python
#
# (c) 2016 -- onwards Georgios Gousios <gousiosg@gmail.com>, Rik Nijessen <riknijessen@gmail.com>
#

from __future__ import print_function

import csv
import argparse
import urllib

from config import *


def load_pr_csv(file):
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

def split(pullreqs, test_split, validation_split):
    train = pullreqs.sample(frac=1-test_split)
    test = pullreqs.drop(train.index)

    train_final = train.sample(frac=1-validation_split)
    validation = train.drop(train_final.index)

    return train_final,validation,test

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', default='default')
parser.add_argument('--balance_ratio', type=float, default=1)
parser.add_argument('--langs', nargs="*", default='')
parser.add_argument('--validation_split', type=float, default=0.1)
parser.add_argument('--test_split', type=float, default=0.2)

args = parser.parse_args()

data = load_pr_csv(ORIG_DATA_FILE)
data = filter_langs(data, args.langs)
data = balance(data, args.balance_ratio)
train, validation, test = split(data, args.test_split, args.validation_split)

print("training length ", len(train))
print("validation length ", len(validation))
print("test length ", len(test))

train.to_csv(train_csv_file % args.prefix, quoting = csv.QUOTE_NONNUMERIC)
validation.to_csv(validation_csv_file % args.prefix, quoting = csv.QUOTE_NONNUMERIC)
test.to_csv(test_csv_file % args.prefix, quoting = csv.QUOTE_NONNUMERIC)
