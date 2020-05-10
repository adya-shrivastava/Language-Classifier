import pickle
import sys

from getAttributes import get_attributes
from DecisionNode import DecisionNode
from DecisionNode import AdaBoostNode

'''
Program used to predict class labels for testing dataset using Adaboost or Decision Tree Models.
'''


def predict(root, test_data):
    """
    Function to predict class label for Decision Tree Algorithm
    :param root: Root node of decision tree
    :param test_data: Input test data
    :return: predicted class label
    """
    if not root.true and not root.false:
        return root.attribute

    if eval(test_data[root.attribute]):
        return predict(root.true, test_data)
    else:
        return predict(root.false, test_data)


def weighted_majority(root, dataset):
    """
    Function to predict expected class label for Adaboost algorithm.
    :param root: Adaboost node
    :param dataset: Input dataset
    :return: predicted class label
    """
    h = root.h
    z = root.z
    N = len(h)

    target = {"en": 0, "nl": 0}
    for i in range(N):
        predictions = predict(h[i], dataset)
        target[predictions] += z[i]

    max_label = None
    max_count = float("-inf")
    for i in target.keys():
        if target[i] > max_count:
            max_label = i
            max_count = target[i]

    return max_label


def main():
    hypothesisFile = sys.argv[1]
    filename = sys.argv[2]
    testing_dataset = []

    file = open(filename, "r", encoding="utf8")
    for line in file.readlines():
        testing_dataset.append(line.rstrip())

    testing_dataset = get_attributes(testing_dataset)

    with open(hypothesisFile, "rb") as output_file:
        tree_root = pickle.load(output_file)

    for test in testing_dataset:
        if isinstance(tree_root, DecisionNode):
            prediction = predict(tree_root, test)

        if isinstance(tree_root, AdaBoostNode):
            prediction = weighted_majority(tree_root, test)

        print(prediction)


if __name__ == '__main__':
    main()