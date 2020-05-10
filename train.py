import pickle
import sys
import adaboost
import decisiontree
from getAttributes import get_attributes
'''
Driver Program to train Decision Tree or Adaboost Models on given input dataset.
'''


def main():
    filename = sys.argv[1]
    hypothesisFile = sys.argv[2]
    learning_type = sys.argv[3]

    input_data = []
    print("opening training file", filename)
    file = open(filename, "r", encoding="utf8")
    for line in file.readlines():
        input_data.append(line.rstrip())

    training_dataset = get_attributes(input_data)

    attribute_list = [i for i in range(len(training_dataset[0]) - 1)]

    if learning_type == "dt":
        print("Calling decisionTree...")
        root = decisiontree.decision_tree(training_dataset, attribute_list, depth=5)
        print("Decision Tree Model ready..")
    elif learning_type == "ada":
        print("Calling Adaboost...")
        root = adaboost.adaboost(training_dataset, attribute_list, K=8)
        print("Adaboost Model ready..")
    else:
        print("Invalid option!")
        sys.exit()

    with open(hypothesisFile, "wb") as output_file:
        pickle.dump(root, output_file)


if __name__ == '__main__':
    main()
