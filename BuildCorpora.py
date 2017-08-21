import os
import sys
import pprint as pp
import spacy
nlp = spacy.load("en")

# Manage imports above this line

def main():
    print("Building files list")
    txt_files_folder = "MalwareTextDB-1.0/data/plaintext/"
    brown_ext_txt_folder = "MalwareTextDB-1.0/data/brown_ext_training_set/"
    txt_files = [txt_files_folder+filename for filename in os.listdir(txt_files_folder)]
    att_files = [brown_ext_txt_folder+filename for filename in os.listdir(brown_ext_txt_folder) if filename.endswith(".att")]
    print(len(txt_files)," files in plaintext")
    print(len(att_files)," files in brown ext training set")
    all_files = txt_files+att_files
    print("Reading from ",len(all_files)," files")
    sentences = []
    for files in all_files:
        buffer_sentences = open(files, encoding="UTF-8").readlines()
        buffer = []
        for sentence in buffer_sentences:
            tokens = nlp(sentence)
            temp_list = []
            temp_list = [token.text for token in tokens]
            buffer.append(" ".join(temp_list))
        sentences.extend(buffer)
    sentences = [x.strip() for x in sentences if x!="\n"]
    print("Writing ",len(sentences)," to file")
    with open("Corpora.txt","w",encoding="UTF-8") as outfile:
        for lines in sentences:
            outfile.write(lines+"\n")
        outfile.close()
    return 0


if __name__ == '__main__':
    status = main()
    sys.exit(status)
