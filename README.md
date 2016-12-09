## pullreqs-dnn

Trying to predict whether a PR will be merged just by examining its diff

### Installing dependencies

This project uses [Keras](https://keras.io). With Keras, you can have multiple
backends, including Tensorflow and Theano. The instructions below are for
Tensorflow.

It is also advisable to use CUDA; installation instructions can be found at
NVIDIA's site

Then, install Tensorflow by exporing *one* of the following variables.

```bash
# Tensorflow without GPU
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl

# Tensorflow with GPU
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
```

Then, install all dependencies

```bash
sudo apt-get install python-pip python-dev libhdf5-dev
sudo pip install --upgrade $TF_BINARY_URL
sudo pip install pandas keras h5py pbr funcsigs
```

### Running

There are two steps to train the network: 1. preprocess the data 2. training.
The idea is that you _tag_ a dataset with a prefix while preprocessing
and use this dataset version for training by applying the same prefix

#### Preprocessing

The preprocessing script downloads (~15GB) the required datasets and transforms them into something the training script can read. It also allows to filter
by language, configure the balancing ratio between merged and unmerged PRs (the
dataset is very unbalanced, as 85% of the PRs are merged on GitHub) and
configure the vocabulary size and sequence lengths to be used for tokenizing.

```bash
./preprocess.py  --help
usage: preprocess.py [-h] [--prefix PREFIX] [--balance_ratio BALANCE_RATIO]
                     [--num_diffs NUM_DIFFS] [--langs [LANGS [LANGS ...]]]
                     [--validation_split VALIDATION_SPLIT]
                     [--vocabulary_size VOCABULARY_SIZE]
                     [--max_sequence_length MAX_SEQUENCE_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX
  --balance_ratio BALANCE_RATIO
  --num_diffs NUM_DIFFS
  --langs [LANGS [LANGS ...]]
  --validation_split VALIDATION_SPLIT
  --vocabulary_size VOCABULARY_SIZE
  --max_sequence_length MAX_SEQUENCE_LENGTH
```

You can run the preprocessing script like this:

```bash
# Only produce data for Ruby projects
./preprocess.py --prefix=ruby --validation_split=0.1 --vocabulary_size 20000 --
max_sequence_length 200
```

#### Training
After preprocessing the data, you'll need to train the model. You can configure
the training method by specifying the number of outputs from the LSTM layer,
the number of outputs from the embeddings layer, the number of epochs to
run and the batch size (number of samples per iteration).

The training is configured to stop if no improvement is seen in validation loss after 5 epochs.

```bash
usage: train.py [-h] [--prefix PREFIX] [--batch_size BATCH_SIZE]
                [--epochs EPOCHS] [--dropout DROPOUT]
                [--lstm_output LSTM_OUTPUT]
                [--embedding_output EMBEDDING_OUTPUT]
                [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --prefix PREFIX
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --dropout DROPOUT
  --lstm_output LSTM_OUTPUT
  --embedding_output EMBEDDING_OUTPUT
  --checkpoint CHECKPOINT
```

```bash
# Train on the previously produced data
./train.py --prefix=ruby --batch_size=256 --epochs=20
```
