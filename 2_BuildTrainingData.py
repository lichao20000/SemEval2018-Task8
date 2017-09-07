import os
import sys
import time
import pprint as pp
import spacy
import csv
from Tools import print_iterable

NLP = spacy.load("en")


def main():
    # Build training data from training files
    # txtcbn_file_folder = "MalwareTextDB-1.0/data/ann+brown/"
    token_file_folder = "MalwareTextDB-1.0/data/tokenized/"
    # files = [txtcbn_file_folder + filename.strip() for filename in os.listdir(txtcbn_file_folder)]
    files = [token_file_folder + filename.strip() for filename in os.listdir(token_file_folder)]
    training_data = []
    print("Found ", len(files), " .token files with training data")
    SENTENCES = []
    LABELS = []
    for file in files:
        training_data.append(("#", file))
        buffer = []
        labels_buffer = []
        for entries in open(file, encoding="UTF-8"):
            try:
                if (entries == "\n"):
                    SENTENCES.append(buffer)
                    LABELS.append(labels_buffer)
                    buffer = []
                    labels_buffer = []
                    continue
                word, _, label = entries.strip().split()
                buffer.append(word)
                labels_buffer.append(label)
            except ValueError:
                print("ValueError ", entries.strip(), " in ", file)
                buffer.append("?")
                labels_buffer.append("O")
                pass
    """
    Quick test to see if both spacy and stanford pos returns the same number of tokens after splitting a sentence
    And they do. Stanford is the baseline used, so not running it again
    print(len(SENTENCES[1]),SENTENCES[1])
    print(len(LABELS[1]),LABELS[1])
    tokens = NLP(" ".join(SENTENCES[1]))
    print(len(tokens)," tokens in Spacy")
    """
    # Parse all training sentences to extract training features into TRAINING_SENTENCES
    TRAINING_SENTENCES = []
    for sentence, label in zip(SENTENCES, LABELS):
        tokens = NLP(" ".join(sentence))
        buffer = []
        counter = 0
        for token, word_label in zip(tokens, label):
            buffer.append((counter, token.text, token.lemma_, token.tag_, token.text[:3], token.text[-3:],len(token.text), word_label))
            counter += 1
        TRAINING_SENTENCES.append(buffer)
    # print(TRAINING_SENTENCES[444])

    # Write TRAINING_SENTENCES to Train.csv
    with open("Train.csv", "w", encoding="UTF-8") as csvout:
        csv_writer = csv.writer(csvout, delimiter="\t")
        for sentence in TRAINING_SENTENCES:
            for feature in sentence:
                csv_writer.writerow(
                    [feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], feature[6],feature[7]])
            csv_writer.writerow(["#EOS"])
        csvout.close()
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
