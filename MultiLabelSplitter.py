import os
import sys


# Manage imports above this line
def split_labels_transform(somelabels):
    labels = []
    boundaries = []
    for alist in somelabels:
        temp1 = []
        temp2 = []
        for entries in alist:
            if (entries.startswith("O")):
                temp1.append("O")
                temp2.append("O")
            else:
                temp1.append(entries.split("-")[0])
                temp2.append(entries.split("-")[1])
        labels.append(temp2)
        boundaries.append(temp1)
    return boundaries, labels


def combine_labels(list1, list2):
    pred_list = []
    sub1 = []
    for x, y in zip(list1, list2):
        sub1 = []
        for x1, y1 in zip(x, y):
            if (x1 == "O"):
                sub1.append(x1)
            elif (x1 != "O"):
                if(y1=="O"):
                    sub1.append("O")
                else:
                    sub1.append(x1+"-"+y1)
        pred_list.append(sub1)
    return pred_list


def main():
    return 0  # End of main


if __name__ == '__main__':
    status = main()
    sys.exit(status)
