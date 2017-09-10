import sys

import numpy as np
import tensorflow as tf

from progressbar import ProgressBar
from nltk import FreqDist
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
from Tools import print_iterable

TRAIN_FILE = "Train.csv"
W2V_EMBEDDINGS = "MalwareText.bin"
PROGRESSBAR = ProgressBar()


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


def vocabularize(training_samples, index_in_tuple):
    return list(set([feature[index_in_tuple] for sentence in training_samples for feature in sentence]))


def substitute_integers(vocab):
    single_map = str.maketrans("0123456789", "X" * 10)
    for i, word in enumerate(vocab):
        try:
            float(word)  # Not sure if this is proper, but it tries to convert to float,
            # if it fails, exception is raised, for normal strings i.e.
            del (vocab[i])
            continue
            # Or replace the digits by X, but it introduces new vocabulary
            vocab[i] = word.translate(single_map)
        except ValueError:
            vocab[i] = word
            continue
    return vocab


def get_embeddings(np_array_of_sentences):
    X = []
    buffer = []
    w2v_model = KeyedVectors.load_word2vec_format(fname=W2V_EMBEDDINGS, binary=True)

    for sentence in np_array_of_sentences:
        for word in sentence:
            try:
                # Word found in embeddings
                embedding = w2v_model[word[1]]
                buffer.append(embedding)
            except KeyError:
                # Word not found in embeddings, so assigning an array of zeros
                embedding = np.zeros(shape=(50), dtype=np.float32)
                buffer.append(embedding)
        X.extend(np.array(buffer))
        buffer = []
    X_len = len(X)
    return np.asarray(X[:int(X_len * 0.8)], dtype=np.float32), np.asarray(X[int(X_len * 0.8):], dtype=np.float32)


def get_sequence_length(nparray_of_sentences):
    sequence_length_vector = []
    for sentence in nparray_of_sentences:
        sequence_length_vector.append(len(sentence))
    return np.array(sequence_length_vector)


def encode_labels(sentence_labels):
    labels = []
    buffer = []
    classes = {}
    for sentence in sentence_labels:
        for label in sentence:
            if (label in classes):
                buffer.append(classes[label])
            else:
                classes[label] = len(classes.keys())
                buffer.append(classes[label])
        labels.extend(np.array(buffer))
        buffer = []
    return np.array(labels), classes


def main():
    print("\nNeural Network approach to TermExtraction")
    # Load the X_train and Y_train from TRAIN_FILE
    print("\nAvailable data")
    x_full, y_full = load_training_file(TRAIN_FILE)
    y_full, classes = encode_labels(y_full)
    print("Classes")
    print(classes.keys(), "\n")
    X_train, X_val = get_embeddings(x_full)
    # print("X_train shape:",X_train.shape)
    # print("X_val shape:",X_val.shape)
    Y_train, Y_val = np.array(y_full[:int(len(y_full) * 0.8)]), np.array(y_full[int(len(y_full) * 0.8):])
    print(len(X_train), " training sentences")
    print(len(Y_train), " labelled training sentences")

    # Get the sequence length of each sentence as 1D tensor
    sequence_length_vector = get_sequence_length(X_train)
    print(len(sequence_length_vector), " sequence_length values")
    print("\nSequence length ", type(sequence_length_vector).__name__, "with shape ", sequence_length_vector.shape)
    print(sequence_length_vector)
    train_sequence_length = np.array(sequence_length_vector[:len(sequence_length_vector) * 80], dtype=np.int32)
    val_sequence_length = np.array(sequence_length_vector[len(sequence_length_vector) * 80:], dtype=np.int32)

    sentence_length_counter = FreqDist([len(sentence) for sentence in X_train])
    print("\nFreqDist of length of sentences and their frequency")
    print(sorted(sentence_length_counter.most_common(100), key=lambda x: x[0], reverse=True))

    # TODO : Construct the RNN model
    """
    Define the RNN model here
    """

    # For logging, used by Tensorboard
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # Parameters
    n_inputs = 50
    n_steps = 1
    n_neurons = 100
    n_outputs = len(classes.keys())  # {B,I}-{Entity,Action,Modifier} and O
    n_layers = 1  # Start of with 1 layer for trial
    learning_rate = 0.001

    # Creating the computation graph
    # seq_length = tf.placeholder(dtype=tf.int32, shape=[None])
    X = tf.placeholder(shape=[None, n_steps, n_inputs], dtype=tf.float32)
    Y = tf.placeholder(shape=[None], dtype=tf.int32)

    # Layers in the computattion graph
    simple_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    # layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
    # multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(cell=simple_cell, inputs=X, dtype=tf.float32)  # , sequence_length=seq_length
    logits = tf.layers.dense(states, n_outputs)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)

    # Metrics ops
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss=loss)
    correct = tf.nn.in_top_k(predictions=logits, targets=Y, k=1)
    accuracy = tf.reduce_mean(tf.cast(x=correct, dtype=tf.float32))

    # Initialise all global variables and the Saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    accuracy_summary = tf.summary.scalar(name="Accuracy", tensor=accuracy)
    file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())

    # Create customized config(ConfigProto)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    # Time to test the graph
    n_epochs = 100

    print("Running Session")
    with tf.Session(config=config) as sess:
        sess.run(init)

        for epoch in PROGRESSBAR(range(n_epochs)):
            # Start saving after an initial test run
            for step in range(0, len(X_train), 500):
                X_train_batch = X_train[step:step + 500].reshape((-1, 1, 50))
                Y_train_batch = Y_train[step:step + 500]
                if (epoch % 50 == 0):  # Checkpoint every 10 epochs, but its a long shot, so save every time
                    summary_str = accuracy_summary.eval(
                        feed_dict={X: X_train.reshape((-1, 1, 50)), Y: Y_train})
                    summary_step = epoch*268+step
                    file_writer.add_summary(summary_str,summary_step)
                sess.run(training_op,
                         feed_dict={X: X_train_batch, Y: Y_train_batch})  # , seq_length: train_sequence_length
        acc_train = accuracy.eval(
            feed_dict={X: X_train.reshape((-1, 1, 50)), Y: Y_train})
        acc_test = accuracy.eval(
            feed_dict={X: X_val.reshape((-1, 1, 50)), Y: Y_val})  # , seq_length: val_sequence_length
        print(epoch, "Train accuracy:", acc_train, " Test accuracy:", acc_test)
        save_path = saver.save(sess=sess, save_path=logdir + "rnn_model_final.ckpt")
    # Run for 100 epochs


    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
