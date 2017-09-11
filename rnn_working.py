import sys

import numpy as np
import tensorflow as tf

from progressbar import ProgressBar
from nltk import FreqDist
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors

TRAIN_FILE = "Train.csv"
TEST_FILE = "Test.csv"
W2V_EMBEDDINGS = "MalwareText.bin"
PROGRESSBAR = ProgressBar()


# Manage imports above this line
def load_file(filename=None, training_file=False):
    sentences = []
    buffer = []
    with open(filename, "r", encoding="UTF-8") as infile:
        if training_file:
            labels = []
            label_buffer = []
            for line in (line for line in infile if line.strip() != ""):
                if (line.strip() == "#EOS"):
                    sentences.append(np.array(buffer, dtype=None))
                    labels.append(np.array(label_buffer, dtype=None))
                    buffer = []
                    label_buffer = []
                    continue
                buffer.append(np.array(line.strip().split("\t")[:-1][1], dtype=None))
                label_buffer.append(np.array(line.strip().split("\t")[-1], dtype=None))
            return np.array(sentences, dtype=None), np.array(labels, dtype=None)
        else:
            for line in (line for line in infile if line.strip() != ""):
                if (line.strip() == "#EOS"):
                    sentences.append(np.array(buffer, dtype=None))
                    buffer = []
                    continue
                buffer.append(np.array(line.strip().split("\t")[:-1][1], dtype=None))
            return np.array(sentences, dtype=None)


def vocabularize(training_samples, index_in_tuple):
    return list(set([feature[index_in_tuple] for sentence in training_samples for feature in sentence]))


def substitute_integers(vocab):
    single_map = str.maketrans("0123456789", "X" * 10)
    for i, word in enumerate(vocab):
        try:
            float(word)
            # Try to convert word to float explicitly
            # If it works, then its a float, so delete it
            del (vocab[i])
            continue
        except ValueError:
            # If it raised ValueError, then do nothing
            vocab[i] = word
            continue
    return vocab


def get_embeddings(np_array_of_sentences=None):
    w2v_model = KeyedVectors.load_word2vec_format(fname=W2V_EMBEDDINGS, binary=True)
    X = []
    buffer = []
    wv_counter = 0
    zeros_counter = 0

    for sentence in np_array_of_sentences:
        for word in sentence:
            try:
                # If word is found in w2v model
                embedding = w2v_model[word]
                wv_counter += 1
                buffer.append(embedding)
            except KeyError:
                zeros_counter += 1
                # Word not found in embeddings, so assigning an array of zeros
                embedding = np.zeros(shape=50, dtype=np.float32)
                buffer.append(embedding)
        if (len(buffer) > 100):
            buffer = []
            continue
        temp = np.zeros((100, 50), dtype=np.float32)
        temp[0:len(buffer)] = buffer
        X.append(np.array(temp))
        buffer = []
    print("Found ", wv_counter, " word vectors and ", zeros_counter, " zeros\n")
    return np.array(X, dtype=np.float32)


def get_sequence_length(nparray_of_sentences=None):
    sequence_length_vector = []
    for sentence in nparray_of_sentences:
        if (len(sentence) < 100):
            sequence_length_vector.append(len(sentence))
    return np.array(sequence_length_vector)


def encode_labels(sentence_labels=None):
    overall_labels = []
    classes = {}
    buffer = []
    for sentence in sentence_labels:
        for label in sentence:
            if (label in classes):
                buffer.append(classes[label])
            else:
                classes[label] = len(classes.keys())
                buffer.append(classes[label])
        if (len(buffer) > 100):
            buffer = []
            continue
        temp = np.zeros(shape=100, dtype=np.int32)
        temp[0:len(buffer)] = buffer
        overall_labels.append(np.array(temp))
        buffer = []
    return np.array(overall_labels), classes


def main():
    print("Neural Network approach to TermExtraction")

    # Load the training data from TRAIN_FILE
    print("\nInput data")
    X, Y = load_file(filename=TRAIN_FILE, training_file=True)
    print(len(X), " training instances(or sentences)")
    print("\nFetching word embeddings")
    seq_len = get_sequence_length(nparray_of_sentences=X)
    X = get_embeddings(np_array_of_sentences=X)
    Y, classes = encode_labels(Y)

    # Display training data stats, for visual inspection
    print("\nTraining data")
    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)
    print("Shape of seq_len : ", seq_len.shape, " --> ", seq_len)
    print("Number of instances in X = ", len(X))
    print("Number of instances in Y = ", len(Y))
    print("Number of classes = ", len(classes))

    # Split the data into train and validation sets
    print("\nSplitting the training data")
    split_size = int(len(X) * 0.8)
    X_train, X_val = X[:split_size], X[split_size:]
    Y_train, Y_val = Y[:split_size], Y[split_size:]
    print("Number of instances in X_train = ", len(X_train))
    print("Number of instances in Y_train = ", len(Y_train))
    print("Number of instances in X_val = ", len(X_val))
    print("Number of instances in Y_val = ", len(Y_val))

    # TODO : RNN model construction
    """
    Define the RNN model here
    """
    # For logging, used by Tensorboard
    now = datetime.utcnow().strftime("%Y-%m-%d_%I-%M-%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # Parameters
    n_inputs = 50
    n_words = 100
    n_neurons = 300
    n_outputs = 100  # {B,I}-{Entity,Action,Modifier} and O
    n_layers = 1  # Start of with 1 layer for trial
    n_batches = 10
    batch_size = len(X) // n_batches
    dropout = 0.2
    learning_rate = 0.001

    # Creating the computation graph
    # The input shape should be something like this
    # batch size, number of words in each sentence, feature vector size (word vector size)
    x = tf.placeholder(shape=[None, n_words, n_inputs], dtype=tf.float32)
    y = tf.placeholder(shape=[None, n_words], dtype=tf.int32)
    # seq_length = tf.placeholder(shape=[None,n_words], dtype=tf.int32)

    # Layers in the computattion graph
    # For single layer NN
    simple_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)  # , state_is_tuple=False # for LSTMcell
    dropout_layer = tf.contrib.rnn.DropoutWrapper(simple_cell,input_keep_prob = dropout)
    # For deep NN
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in
              range(n_layers)]  # , state_is_tuple=False # for LSTMCell
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers,state_is_tuple=False)

    # Hidden layer level
    outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=x, dtype=tf.float32,
                                        time_major=True)  # , sequence_length=seq_length

    # Dense output level
    logits = tf.layers.dense(states, n_outputs, activation=tf.nn.relu)

    # Classifier level
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[0, :], logits=logits)

    # Metrics ops
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss=loss)
    correct = tf.nn.in_top_k(predictions=logits, targets=y[0, :], k=1)
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

        # Run for (n_epoch) epochs
        for epoch in PROGRESSBAR(range(n_epochs)):
            # Start saving after an initial test run
            for step in range(0, len(X_train), batch_size):
                X_next_batch = X_train[step:step + batch_size]  # .reshape((batch_size, n_words, 50))
                Y_next_batch = Y_train[step:step + batch_size]  # .reshape((batch_size, n_words))
                # seq_len_batch = seq_len[step:step + batch_size].reshape(batch_size,n_words)
                if (epoch % 50 == 0):  # Checkpoint every 10 epochs, but its a long shot, so save every time
                    summary_str = accuracy_summary.eval(
                        feed_dict={x: X_next_batch, y: Y_next_batch})
                    summary_step = epoch * n_batches + step
                    file_writer.add_summary(summary_str, summary_step)

                sess.run(training_op,
                         feed_dict={x: X_next_batch, y: Y_next_batch})  # , seq_length: train_sequence_length
            acc_train = accuracy.eval(feed_dict={x: X_train, y: Y_train})
            acc_test = accuracy.eval(
                feed_dict={x: X_val, y: Y_val})  # , seq_length: val_sequence_length
            print("\nEpoch : ",epoch," Train accuracy:", acc_train, " Test accuracy:", acc_test)
            #if(acc_train>0.80 and acc_test>0.80):
            #    break
        save_path = saver.save(sess=sess, save_path=logdir + "rnn_model_final.ckpt")
        sess.close()
        exit(0)
        # Time to predict on our test data
        print("Predicting on test data")
        X_test = load_training_file(filename=TEST_FILE, training_file=False)
        X_test, _ = get_embeddings(X_test)
        results = sess.run(logits, feed_dict={X: X_test.reshape((len(X_test), 1, 50))})
        print(len(results), " words in test data")
        # for word in results:
        #    print(len(word), "\t", max(word) == word[0], "\t", word)
        print(classes)
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
