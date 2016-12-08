import pandas as pd
import time

## Files
ORIG_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/data.csv"
ORIG_DATA_FILE = "pr-data.csv"
DIFFS_DATA_URL = "https://dl.dropboxusercontent.com/u/57978013/pullreq-patches.tar.gz"
DIFFS_DIR = "pullreq-patches"
DIFFS_FILE = "pullreq-patches.tar.gz"
DATASETS_DIR = "datasets"

pd.set_option('display.max_rows', 10)


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
              (f.__name__, te - ts)
        return result

    return timed
