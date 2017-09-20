import os
import sys
import pprint as pp
import re
from nltk import ngrams
from nltk.corpus import stopwords


# Manage imports above this line

def main():
    # Read all tokens from Task1and2-input file and build sentences
    test1_input = "SemEval_input/Task1and2-input"
    with open(test1_input, encoding="UTF-8") as input_file:
        entries = [x.strip() for x in input_file.readlines()]
    print(len(entries), " tokens")
    sentences = []
    buffer = []
    for i, token in enumerate(entries):
        if (token == "." and entries[i + 1] == ""):
            buffer.append(token)
            sentences.append(buffer)
            buffer = []
        else:
            buffer.append(token)
    print(len(sentences), " sentences")

    # Read Annotation labels file and retrieve the key phrases
    # and then process them into keywords
    with open("MalwareTextDB-1.0/annotation_guidelines/Attribute Labels.csv") as labels_file:
        entries = [x.strip().split(",") for x in labels_file.readlines()]
        labels_file.close()
    keyphrases = []
    stopword = stopwords.words("english")
    for rows in entries[1:]:
        keyphrases.append(rows[-1].strip())
    keyphrases = [x for x in keyphrases if x != ""]
    keywords = []
    for entry in keyphrases:
        keywords.extend([x for x in re.findall(r"[\w ]*", entry) if x != ""])
    keywords = set(keywords)
    print(len(keywords), " keywords")

    exit(0)
    # For each sentence build ngrams of multiple order and
    # check for keyword match
    for i, sentence in enumerate(sentences):
        sentence = [x.lower() for x in sentence]
        tokens = []
        sentence_as_string = " ".join(sentence)
        print(sentence_as_string)
        tokens.extend(list(ngrams(sentence_as_string.split(), n=1)))
        tokens.extend(list(ngrams(sentence_as_string.split(), n=2)))
        tokens.extend(list(ngrams(sentence_as_string.split(), n=3)))
        tokens = set([" ".join(x) for x in tokens if x not in stopword])
        for ngram in tokens:
            if (ngram in keywords):
                print(i, ngram)
                pass

    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
