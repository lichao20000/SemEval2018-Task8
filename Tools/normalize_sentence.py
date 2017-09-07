import os
import sys


# Manage imports above this line

def normalize(input_sentence):
    print(input_sentence)
    return 0  # End of main


if __name__ == '__main__':
    sentence = " ".join(sys.argv[1:])
    print(sentence)
    sys.exit(0)
