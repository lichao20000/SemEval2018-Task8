from __future__ import print_function
import os
import sys
import pprint as pp
import spacy

nlp = spacy.load("en")


# TODO : Identify Subject and Object in a sentence, more specifically, identify the Subject, everything else are objects
# !!! The Subject must be as part of the WHOLE SENTENCE, not just in the labelled tokens
# achieving the above would make the task, an ease

# Manage imports above this line

def main():
    # Read the filesnames from doclist.txt and append .rel to the end
    # and build the input files list for task 3
    inputfiles = open("SemEval_input/Task3-input/doclist.txt").readlines()
    inputfiles = ["SemEval_input/Task3-input/" + x.strip() + ".rel" for x in inputfiles if x != "" and x != "\n"]

    sentences = []
    for files in inputfiles:
        print(files)
        buffer = []
        bufferSentence = ""
        for tokens in open(files, encoding="UTF-8"):
            if (tokens.strip() == "" or tokens.strip() == "\n"):
                continue
            token, pos, label, rel_id = tokens.strip().split(" ")
            bufferSentence += " " + token
            buffer.append((token, pos, label, rel_id))
            if (token == pos == "." and label == "O"):
                buffer.append((bufferSentence.strip()))
                sentences.append(buffer)
                buffer = []
                bufferSentence = ""
        buffer.append((bufferSentence))
        sentences.append(buffer)
    print(len(sentences), " sentences from ", len(inputfiles), " files")

    # Extract entity pairs from the input sentences
    label_phrase = []
    for sentence in sentences:
        combinations = []
        label = ""
        buffer = ""
        for tokens in sentence[:-1]:
            if (tokens[0] == ","):
                continue
            if (tokens[2] == "O"):
                if (buffer != ""):
                    combinations.append((label, buffer.strip()))
                    buffer = ""
                    label = ""
                    continue
                else:
                    continue
            elif (tokens[2] != "O"):
                if (buffer == ""):
                    buffer += " " + tokens[0]
                    label = tokens[2][2:]
                else:
                    if (tokens[2][2:] == label):
                        buffer += " " + tokens[0]
                    else:
                        combinations.append((label, buffer.strip()))
                        label = tokens[2][2:]
                        buffer = tokens[0]
        if (buffer != "" and label != ""):
            combinations.append((label, buffer.strip()))
        if (len(combinations) > 0):
            label_phrase.append(combinations)
    # # pp.pprint(label_phrase)
    # # print("\n", len(label_phrase), " token pairs")

    # Play with the entity pairs, test some theories
    task3_out = []
    for j, entries in enumerate(label_phrase):
        print("Relevant sentence ", j, " with ", len(entries), " entities")
        for i, x in enumerate(entries):
            print(0, x[0] + str(i + 1))
    """
    task3_out = []
    for list_entries in label_phrase:
        temp = []
        for i, x in enumerate(list_entries):
            for j, y in enumerate(list_entries):
                if (i == j):
                    temp.append((0,j+1))
                    continue
                temp.append((i+1,j+1))
        task3_out.append(temp)
    task3_out = [[x for x in sorted(y,key=lambda x:x[0])]for y in task3_out]
    with open("Entity_pair_sample.txt","w",encoding="UTF-8") as outfile:
        for entries in task3_out:
            for pairs in entries:
                outfile.write(str(pairs[0])+"\t"+str(pairs[1])+"\n")
            outfile.write("\n")
        outfile.close()
    """
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
