from __future__ import print_function
import os
import sys
import pprint as pp
import spacy

nlp = spacy.load("en")


# Manage imports above this line

# TODO : Identify Subject and Object in a sentence, more specifically, identify the Subject, everything else are objects
# !!! The Subject must be as part of the WHOLE SENTENCE, not just in the labelled tokens
# Incase there are multiple subjects, it is the task of previous task(!) to identify multiple subjects as a single subject
# achieving the above would make the task, an ease

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
        combinations.append((0, sentence[-1]))
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
    print(len(label_phrase), " relevant sentences")
    pp.pprint(label_phrase)
    # Convert the entity pairs into the output format, also the processing format
    task3_out = []
    for j, entries in enumerate(label_phrase):
        temp = []
        temp.append(entries[0])
        for a, x in enumerate(entries[1:]):
            temp.append((0, a + 1, "TOKEN0," + x[0], "O"))
            for b, y in enumerate(entries[1:]):
                if (a == b):
                    continue
                else:
                    temp.append((a + 1, b + 1, x[0] + "," + y[0], "O"))
        task3_out.append(temp)
    task3_out = [[x for x in sorted(y, key=lambda x: x[0])] for y in task3_out]

    # Test some rules here for relation prediction
    # RULES RULES RULES
    # Maintain a sublist of labelled entities for each relevant sentence in task3_out
    for i, sentences in enumerate(task3_out):
        sublist = [x[2].split(",")[1] for x in sentences[1:] if x[0] == 0]
        pp.pprint(sublist)
        for j, pair in enumerate(sentences[1:]):
            id1, id2, entity_pair, relation = pair
            entity1, entity2 = entity_pair.split(",")
            if (entity1 == "Modifier" and entity2 == "Entity" and int(id1) == int(id2) - 1):
                relation = "ModObj"
                task3_out[i][j] = (id1, id2, entity_pair, relation)
            elif (entity1 == "Entity" and entity2 == "Action" and int(id1) < int(id2) and (
                        int(id2) - int(id1)) in range(0, 2)):
                relation = "SubjAction"
                task3_out[i][j] = (id1, id2, entity_pair, relation)
            elif (entity1 == "Action"):
                for tempindex, subpair in enumerate(sentences[j:]):
                    subid1, subid2, entity_entity_subpair, _ = subpair
                    subentity1, subentity2 = entity_entity_subpair.split(",")
                    if (subentity1 != entity1):
                        break
                    if (subentity2 == "Entity" and abs(int(subid2) - int(id1)) < 2):
                        relation = "ActionObj"
                        task3_out[i][j + tempindex] = (subid1, subid2, entity_entity_subpair, relation)
                        break
                    elif (subentity2 == "Action"):
                        pass
                    elif (subentity2 == "Modifier" and int(id1) < int(subid2) and abs(int(subid2) - int(id1)) < 2):
                        relation = "ActionMod"
                        task3_out[i][j + tempindex] = (subid1, subid2, entity_entity_subpair, relation)
                        break

    # Write the entityid, entityid, entity pair to file for visual inspection
    with open("Entity_pair_sample.txt", "w", encoding="UTF-8") as outfile:
        for lists in task3_out:
            for entry in lists:
                if (len(entry) == 2):
                    outfile.write(entry[1] + "\n")
                    continue
                outfile.write(str(entry[0]) + "\t" + str(entry[1]) + "\t" + entry[2] + "\t" + entry[3] + "\n")
            outfile.write("\n")

    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
