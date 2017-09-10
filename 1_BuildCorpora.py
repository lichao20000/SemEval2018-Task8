import os
import sys
import pprint as pp
import spacy
from gensim.models import Word2Vec
from Tools import print_iterable

NLP = spacy.load("en")

# Manage imports above this line
def process_sentences(sentences_list):
    processsed_sentences_list = []
    for sentence in sentences_list:
        if (sentence == ""):
            continue
        if (sentence[0] in "0123456789"):
            continue
        if (sentence.split()[0].istitle()):
            processsed_sentences_list.append(sentence)
    return processsed_sentences_list


def main():
    # Create a list of all the files, from which to retrieve the training data
    print("Building files list")
    txt_files_folder = "MalwareTextDB-1.0/data/plaintext/"
    brown_ext_txt_folder = "MalwareTextDB-1.0/data/brown_ext_training_set/"
    txt_files = [txt_files_folder + filename for filename in os.listdir(txt_files_folder)]
    att_files = [brown_ext_txt_folder + filename for filename in os.listdir(brown_ext_txt_folder) if
                 filename.endswith(".att")]
    print(len(txt_files), " files in plaintext")
    print(len(att_files), " files in brown ext training set")
    all_files = txt_files + att_files

    # Read all labelled sentences from the training files
    print("Reading from ", len(all_files), " files")
    SENTENCES = []
    for files in all_files:
        buffer_SENTENCES = open(files, encoding="UTF-8").readlines()
        buffer = []
        for sentence in buffer_SENTENCES:
            tokens = NLP(sentence)
            temp_list = [token.text for token in tokens]
            buffer.append(" ".join(temp_list))
        SENTENCES.extend(buffer)
    SENTENCES = [x.strip() for x in SENTENCES if x != "\n"]
    SENTENCES = process_sentences(SENTENCES)

    print("Writing ", len(SENTENCES), " to file")
    with open("Corpora.txt", "w", encoding="UTF-8") as outfile:
        for lines in SENTENCES:
            outfile.write(lines + "\n")
        outfile.close()
    print("Creating Word2Vec model from ", len(SENTENCES), " SENTENCES")
    SENTENCES = [sentence.split() for sentence in SENTENCES]
    model = Word2Vec(SENTENCES, size=50, window=25, min_count=5, workers=16, seed=42)
    print("Saving model")
    model.wv.save_word2vec_format("MalwareText.bin", binary=True)
    return 0


if __name__ == '__main__':
    status = main()
    sys.exit(status)
