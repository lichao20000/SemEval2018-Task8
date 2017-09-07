from __future__ import print_function
import os
import sys


# Manage imports above this line

def convertToBILOU(BIOlist):
    bilouLabels = []
    for i, sublist in enumerate(BIOlist):
        buffer = []
        for j, entries in enumerate(sublist):
            if (entries == "O"):
                buffer.append("O")
                continue
            if (entries[0] == "B"):
                if (sublist[j + 1][0] == "I"):
                    buffer.append("B" + entries[1:])
                    continue
                elif (sublist[j + 1][0] == "O" or sublist[j + 1][0] == "B"):
                    buffer.append("U" + entries[1:])
                    continue
            if (entries[0] == "I"):
                if (sublist[j + 1][0] == "O" or sublist[j + 1][0] == "B"):
                    buffer.append("L" + entries[1:])
                    continue
                else:
                    buffer.append("I" + entries[1:])
                    continue
        bilouLabels.append(buffer)
    return bilouLabels


def convertToBIO(BILOUlist):
    bioLabels = []

    for i, sublist in enumerate(BILOUlist):
        buffer = []
        for j, entry in enumerate(sublist):
            if (entry[0] == "O"):
                buffer.append("O" + entry[1:])
                continue
            if (entry[0] == "U"):
                buffer.append("B" + entry[1:])
                continue
            if (entry[0] == "B"):
                buffer.append("B" + entry[1:])
                continue
            if (entry[0] == "I"):
                buffer.append("I" + entry[1:])
                continue
            if (entry[0] == "L"):
                buffer.append("I" + entry[1:])
                continue
        bioLabels.append(buffer)
    return bioLabels
