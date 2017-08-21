import sys

# Manage imports above this line

def main():
    task1and2_input = open("SemEval_input/Task1and2-input",encoding="UTF-8")
    for lines in task1and2_input:
        print(lines)
    return 0


if __name__ == '__main__':
    status = main()
    sys.exit(status)