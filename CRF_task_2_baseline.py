import os
import sys
import matplotlib.pyplot as plt
import time
import pprint as pp

plt.style.use('ggplot')

import sklearn_crfsuite
from sklearn_crfsuite import metrics


# Manage imports above this line

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        # 'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    # return [label for token, postag, onehot, label in sent]
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    # return [token for token, postag, onehot, label in sent]
    return [token for token, postag, label in sent]


def main():
    # Build training data from training files
    # txtcbn_file_folder = "MalwareTextDB-1.0/data/ann+brown/"
    token_file_folder = "MalwareTextDB-1.0/data/tokenized/"
    # files = [txtcbn_file_folder + filename.strip() for filename in os.listdir(txtcbn_file_folder)]
    files = [token_file_folder + filename.strip() for filename in os.listdir(token_file_folder)]
    training_data = []
    # print("Found ", len(files), " .txtcbn files with training data")
    print("Found ", len(files), " .token files with training data")
    training_sentences = []
    for file in files:
        training_data.append(("#", file))
        temp = []
        for entries in open(file, encoding="UTF-8"):
            if (entries.strip().split(" ")[0] == "" and len(entries.split(" ")) == 1):
                continue
            elif (len(entries.strip(" ")) == 3):
                print(entries)
                continue
            training_data.append((entries.strip().split(" ")))
            temp.append((entries.strip().split(" ")))
            # if (temp[-1][0] == "." and temp[-1][3] == "O"):
            if (temp[-1][0] == "." and temp[-1][2] == "O"):
                training_sentences.append(temp)
                temp = []
    print(len(training_data), " samples in training data")
    print(len(training_sentences), " sentences ")
    # training_sentences = [[x for x in y if len(x) == 4] for y in training_sentences]
    training_sentences = [[x for x in y if len(x) == 3] for y in training_sentences]

    # Write training data, for visual purposes
    with open("Tokens.tsv", "w", encoding="UTF-8") as tsvout:
        for entries in training_data:
            if (entries[0] == "#"):
                tsvout.write(entries[0] + "\t" + entries[1] + "\n")
                continue
            elif (entries[0] == ""):
                tsvout.write("\n")
            elif (len(entries) == 4):
                tsvout.write(entries[0] + "\t" + entries[1] + "\t" + entries[2] + "\t" + entries[3] + "\n")
            else:
                continue
        tsvout.close()

    # Build training data for CRF
    X_train = [sent2features(sentence) for sentence in training_sentences[:int(len(training_sentences) * .80)]]
    Y_train = [sent2labels(sentence) for sentence in training_sentences[:int(len(training_sentences) * .80)]]
    X_test = [sent2features(sentence) for sentence in training_sentences[int(len(training_sentences) * .80):]]
    Y_test = [sent2labels(sentence) for sentence in training_sentences[int(len(training_sentences) * .80):]]

    # Build CRF and fit train data
    print("Fit training data")
    start_time = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )  # verbose=True
    crf.fit(X_train, Y_train)

    # Evaluation
    labels = list(crf.classes_)
    labels.remove('O')
    print(labels)
    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(Y_test, y_pred,
                          average='weighted', labels=labels)
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        Y_test, y_pred, labels=sorted_labels, digits=3
    ))
    print("Running time : ", time.time() - start_time)

    results = crf.predict(X_test)
    print("Writing results to file")
    outfile = open("Predict_out.txt", "w", encoding="UTF-8")
    for x, y in zip([[x[0] for x in y] for y in training_sentences[int(len(training_sentences) * .80):]], results):
        for x1, y1 in zip(x, y):
            outfile.write(x1 + "\t" + y1 + "\n")
    outfile.close()

    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
