import math
import decisiontree
from DecisionNode import AdaBoostNode


def adaboost(dataset, attribute_list, K=8):
    """
    Function used to train the adaboost model
    :param dataset: input dataset to train the Adaboost Model
    :param attribute_list: list of attributes present
    :param K: Number of decision stumps used. Default value = 8
    :return: trained Adaboost model with hypothesis and weights
    """
    N = len(dataset)
    w = [1/N] * N

    h = []
    z = []
    eps = 0.00001
    for k in range(0, K):
        root = decisiontree.decision_tree(dataset, attribute_list, old_dataset=[], depth=1, w=w)
        h.append(root)
        attribute = root.attribute
        error = 0
        for j in range(0, N):
            if _predict(root, dataset[j], attribute) != dataset[j][-1]:
                error += w[j]

        if error == 0:
            error += eps

        for j in range(0, N):
            if _predict(root, dataset[j], attribute) == dataset[j][-1]:
                w[j] *= (error / (1-error))

        w = normalize(w)
        z.append(math.log((1-error)/error))

    return AdaBoostNode(h, z)


def normalize(w):
    """
    Function to normalize the weights
    :param w: weights
    :return: normalized weights
    """
    sum = 0
    for i in range(len(w)):
        sum += w[i]

    for i in range(len(w)):
        w[i] = w[i] / sum

    return w


def _predict(root, dataset, attrib):
    """
    Predicts the expected label for a row in dataset
    :param root: root of the Decision Stump
    :param dataset: input dataset
    :param attrib: attribute value of the root of tree
    :return: expected class label
    """
    if not root.true and not root.false:
        return root.attribute

    if eval(dataset[attrib]):
        return _predict(root.true, dataset, attrib)

    else:
        return _predict(root.false, dataset, attrib)
