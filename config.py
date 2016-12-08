import pandas as pd
import time
import os

## Files
ORIG_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/data.csv"
ORIG_DATA_FILE = "pr-data.csv"
DIFFS_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/pullreq-patches.tar.gz"
DIFFS_DIR = "pullreq-patches"
DIFFS_FILE = "pullreq-patches.tar.gz"
DATASETS_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"

pd.set_option('display.max_rows', 10)

if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

x_train_file = os.path.join(DATASETS_DIR, 'x_train_%s.pcl')
y_train_file = os.path.join(DATASETS_DIR, 'y_train_%s.pcl')
x_val_file = os.path.join(DATASETS_DIR, 'x_val_%s.pcl')
y_val_file = os.path.join(DATASETS_DIR, 'y_val_%s.pcl')
config_file = os.path.join(DATASETS_DIR, 'config_%s.pcl')
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_%s.{epoch:02d}-{val_loss:.2f}.hdf5')

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
              (f.__name__, te - ts)
        return result

    return timed
