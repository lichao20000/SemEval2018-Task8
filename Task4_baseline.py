import os
import sys
import pprint as pp
import re
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

sw = stopwords.words("english")
lem = WordNetLemmatizer()
ps = PorterStemmer()
nlp = spacy.load("en")


# Manage imports above this line
# Implement get_context_features (DONE)
# and include it as part of get_attribute_id (DONE)
def get_context_features(a_sentence):
    tokens = nlp(a_sentence)
    features = []
    for token in tokens:
        if (token.tag_.startswith("V") or token.tag_.startswith("N") and token.lemma_ not in sw):
            try:
                features.append(re.match("^[a-zA-Z]*$", token.lemma_).group(0))
            except AttributeError:
                pass
    return features


def get_attribute_id(phrase, bucket, contextual_features):
    id = "O"
    phrase = lem.lemmatize(phrase, "v")
    global_max = 0
    for item in bucket:
        relevance_count = 0
        for keywords in item[3]:
            if (keywords in contextual_features):
                relevance_count += 1
        if (relevance_count > global_max):
            id = str(item[0])
            global_max = relevance_count
    return id


def main():
    # Read task 4 doclist.txt and get all filenames to process
    task4_input_folder = "SemEval_input/Task4-input/"
    task4_input_files = open("SemEval_input/Task4-input/doclist.txt", encoding="UTF-8").readlines()
    task4_input_files = [task4_input_folder + x.strip() + ".in" for x in task4_input_files]

    # Read the lines from the input files and process it
    # MAINTAIN FILE ORDER and SENTENCE ORDER
    sentences = []
    annotations = []  # Or only the Actions, not sure yet
    for files in task4_input_files:
        with open(files, "r") as infile:
            newline = True
            temp = []
            for line in infile:
                if (line == "\n"):
                    annotations.append(sorted(temp, key=lambda x: x[2]))
                    temp = []
                    newline = True
                elif (newline):
                    sentences.append(line.strip())
                    newline = False
                else:
                    if (line.startswith("R")):
                        continue
                    id, token_pair, phrase = line.strip().split("\t")
                    token, start, end = token_pair.split(" ")
                    temp.append((id, token, int(start), int(end), phrase))
    print("Processing input files")
    print(len(sentences), " sentences")
    print(len(annotations), " corresponding annotation lists")
    print("\n")

    # Load the attributes file and categorise them into 4 lists
    # The attribute labels file has been converted from PDF->Excel->CSV
    attributes_file = "MalwareTextDB-1.0/annotation_guidelines/Attribute Labels.csv"
    attributes_list = open(attributes_file, encoding="UTF-8").readlines()
    # Now that we have all the contents, process them into 4 buckets
    attributes_list = [x.strip().split(",") for x in attributes_list]
    ActionName = []
    Capability = []
    StrategicObjectives = []
    TacticalObjectives = []
    currentBucket = ""
    for entries in attributes_list:
        if (len(entries) == 1):
            currentBucket = entries[0]
        elif (entries[0] == entries[1] == entries[2] == entries[3] == ""):
            pass
        elif (currentBucket == "ActionName"):
            _, att_no, att_name, att_desc, keyword_pair = entries
            keywords = re.findall("[A-Za-z]*", keyword_pair)
            keywords = [x.lower() for x in keywords if x != ""]
            keywords = list(set(keywords))
            ActionName.append((int(att_no), att_name, att_desc, keywords))

        elif (currentBucket == "Capability"):
            _, att_no, att_name, att_desc, keyword_pair = entries
            keywords = re.findall("[A-Za-z]*", keyword_pair)
            keywords = [x.lower() for x in keywords if x != ""]
            keywords = list(set(keywords))
            Capability.append((int(att_no), att_name, att_desc, keywords))

        elif (currentBucket == "StrategicObjectives"):
            _, att_no, att_name, att_desc, keyword_pair = entries
            keywords = re.findall("[A-Za-z]*", keyword_pair)
            keywords = [x.lower() for x in keywords if x != ""]
            keywords = list(set(keywords))
            StrategicObjectives.append((int(att_no), att_name, att_desc, keywords))

        elif (currentBucket == "TacticalObjectives"):
            _, att_no, att_name, att_desc, keyword_pair = entries
            keywords = re.findall("[A-Za-z]*", keyword_pair)
            keywords = [x.lower() for x in keywords if x != ""]
            keywords = list(set(keywords))
            TacticalObjectives.append((int(att_no), att_name, att_desc, keywords))
    # Extract only  the keywords into subbuckets
    action_names = []
    for i in ActionName:
        action_names.extend(i[3])
    action_names = set(action_names)
    capability_names = []
    for i in Capability:
        capability_names.extend(i[3])
    capability_names = set(capability_names)
    strategic_objectives_names = []
    for i in StrategicObjectives:
        strategic_objectives_names.extend(i[3])
    strategic_objectives_names = set(strategic_objectives_names)
    tactical_objectives_names = []
    for i in TacticalObjectives:
        tactical_objectives_names.extend(i[3])
    tactical_objectives_names = set(tactical_objectives_names)

    # Just visualize the buckets
    print("Processing AttributeLabels file")
    print(len(ActionName), " ActionName descriptions and ", len(action_names), " keywords")
    print(len(Capability), " Capability descriptions and ", len(capability_names), " keywords")
    print(len(StrategicObjectives), " StrategicObjectives descriptions and ", len(strategic_objectives_names),
          " keywords")
    print(len(TacticalObjectives), " TacticalObjectives descriptions and ", len(tactical_objectives_names), " keywords")
    print("\n")

    # Simple processing of each sentence
    for Sentence, Annotations in zip(sentences, annotations):
        for Annotation in Annotations:
            if (Annotation[1] == "Action"):
                action_name_id = get_attribute_id(Annotation[4], ActionName, get_context_features(Sentence))
                capability_id = get_attribute_id(Annotation[4], Capability, get_context_features(Sentence))
                strategic_objective_id = get_attribute_id(Annotation[4], StrategicObjectives,
                                                          get_context_features(Sentence))
                tactical_objective_id = get_attribute_id(Annotation[4], TacticalObjectives,
                                                         get_context_features(Sentence))
                try:
                    print(Sentence)
                    print(Annotation[0] + "\t" + Annotation[
                        4] + "\t" + action_name_id + "\t" + capability_id + "\t" + strategic_objective_id + "\t" + tactical_objective_id)
                except ValueError:
                    pass
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
