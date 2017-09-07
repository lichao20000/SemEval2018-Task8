import os
import csv
import sys
import spacy
from Tools import print_iterable

NLP = spacy.load("en")
INPUT_FILE = "SemEval_input/Task1and2-input"


# Manage imports above this line

def main():
    """
    Returns: 0 on succesfully building the testing file.
    Else raises some error and dies tragically in an exception accident

    """
    """ 
    Load all words from the Task1and2-input file to process them into sentences
    """
    wordslist = open(INPUT_FILE, encoding="UTF-8").readlines()
    SENTENCES = []
    buffer = []

    # Extract whole sentences from file
    for word in wordslist:
        if (word == "\n"):
            SENTENCES.append(buffer)
            buffer = []
            continue
        buffer.append(word.strip())
    # print_iterable.print_iterable(SENTENCES[:1])

    """
    Process each sentence to get the
    lemma,prefix,suffix,pos tag etc...features
    into the TESTING_SENTENCES attribute
    """
    TESTING_SENTENCES = []
    for sentence in SENTENCES:
        buffer = []
        tokens = NLP(" ".join(sentence))
        for i, token in enumerate(tokens):
            buffer.append((i, token.text, token.lemma_, token.tag_, token.text[:3], token.text[-3:], token.dep_,len(token.text)))
        TESTING_SENTENCES.append(buffer)

    """
    Write TESTING_SENTENCES to file, which will be used 
    to test a ANN    
    """
    with open("Test.csv", "w", encoding="UTF-8") as csvout:
        csv_writer = csv.writer(csvout, delimiter="\t", )
        for sentence in TESTING_SENTENCES:
            for feature in sentence:
                csv_writer.writerow([feature[0], feature[1], feature[2], feature[3], feature[4], feature[5],feature[6]])
            csv_writer.writerow(["#EOS"])
        csvout.close()
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
