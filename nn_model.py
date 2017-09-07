import os
import sys
import pprint as pp
import tensorflow
from Tools import print_iterable
import numpy as np

TRAIN_FILE = "Train.csv"


# Manage imports above this line
def load_training_file(filename):
    SENTENCES = []
    LABELS = []
    buffer = []
    label_buffer = []
    with open(filename, "r", encoding="UTF-8") as infile:
        for line in (line for line in infile if line.strip() != ""):
            if (line.strip() == "#EOS"):
                SENTENCES.append(np.array(buffer, dtype=None))
                LABELS.append(np.array(label_buffer, dtype=None))
                buffer = []
                label_buffer = []
                continue
            buffer.append(np.array(line.strip().split("\t")[:-1], dtype=None))
            label_buffer.append(np.array(line.strip().split("\t")[-1], dtype=None))

    return np.array(SENTENCES, dtype=None), np.array(LABELS, dtype=None)


def main():
    # Load the X_train and Y_train from TRAIN_FILE
    X_train, Y_train = load_training_file(TRAIN_FILE)
    print(len(X_train), " training sentences")
    print(len(Y_train), " labelled training sentences")
    pp.pprint(X_train[:5])
    pp.pprint(Y_train[:5])
    # TODO: PREPROCESSING and NN CONSTRUCTION PHASE

    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
