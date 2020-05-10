import math

from DecisionNode import DecisionNode


def same_class(dataset):
    """
    Returns True if all the samples in the dataset belong to the same class.
    False otherwise.
    :param dataset: input dataset
    :return: True or False
    """
    value = set()
    for row in range(0, len(dataset)):
        value.add(dataset[row][-1])
    return len(value) == 1


def init_entropy(dataset, w=None):
    """
    Calculates entropy with respect to class labels
    :param dataset: Input dataset
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: calculated entropy
    """
    count_en = 0
    count_nl = 0
    for i in range(len(dataset)):
        c = 1 if w is None else w[i]
        if dataset[i][-1] == "en":
            count_en += c
        else:
            count_nl += c

    p = count_en / len(dataset)
    q = count_nl / len(dataset)
    # q = 1 - p
    entropy = p * math.log2(p) + q * math.log2(q)
    return entropy * (-1)


def get_current_entropy(dataset, attribute, w=None):
    """
    Returns entropy of a given attribute.
    :param dataset: Input Dataset
    :param attribute: Current attribute
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: calculated entropy
    """
    a = {"True": 0, "False": 0}
    b = {"True": 0, "False": 0}
    for i in range(len(dataset)):
        c = 1 if w is None else w[i]
        if dataset[i][-1] == "en":
            a[dataset[i][attribute]] += c
        else:
            b[dataset[i][attribute]] += c

    total = {"True": a["True"] + b["True"],
             "False": a["False"] + b["False"]}

    a["entropy"] = entropy(a, total, "True")
    b["entropy"] = entropy(b, total, "False")

    return a, b


def entropy(current_label, total, flag):
    """
    Helper function to calculate the entropy
    """
    if total[flag] != 0:
        p = current_label[flag] / total[flag]
    else:
        p = 0
    q = 1 - p

    if p == 0 or p == 1:
        entropy = 0
    else:
        entropy = p * math.log2(p) + q * math.log2(q)

    return (-1) * entropy


def information_gain(dataset, attribute, initial_entropy, w=None):
    """
    Calculates the information gain of an attribute
    :param dataset: Input dataset
    :param attribute: current attribute
    :param initial_entropy: entropy with respect to class label
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: information gain for the given attribute
    """

    a, b = get_current_entropy(dataset, attribute, w)
    info_gain = (((a["True"] + b["True"]) / len(dataset)) * a["entropy"]) \
                + (((a["False"] + b["False"]) / len(dataset)) * b["entropy"])

    return initial_entropy - info_gain


def get_best_split(dataset, attributes, w=None):
    """
    Returns the attribute with maximum gain
    :param dataset: Input dataset
    :param attributes: list of all attributes
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: attribute with the maximum gain
    """
    max_info_gain = 0
    best_attribute = None
    initial_entropy = init_entropy(dataset, w)
    for i in range(len(attributes)):
        current_gain = information_gain(dataset, attributes[i], initial_entropy, w)

        if current_gain > max_info_gain:
            best_attribute = attributes[i]
            max_info_gain = current_gain

    return best_attribute


def get_count_of_max_label(dataset, w=None):
    """
    Returns class label which occurs the most
    :param dataset: Inpute dataset
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: class label. If count of both labels are equal, the function will arbitrarily return label 'en'
    """

    label = {"en": 0, "nl": 0}
    for i in range(len(dataset)):
        val = dataset[i][-1]
        weight = w[i] if w is not None else 1
        label[val] += weight

    return "en" if label["en"] > label["nl"] else "nl"


def decision_tree(dataset, attributes, old_dataset=[], depth=5, w=None):
    """
    Function to train a decision tree model using input extracted attributes from the
    input dataset.
    :param dataset: Input dataset.
    :param attributes: List of attributes.
    :param old_dataset: Parent dataset.
    :param depth: maximum depth for decision tree. Default is 5 for decision tree
    :param w: weights of the sample. Default is None for Decision Tree algorithm.
    :return: Root node of the trained decision tree.
    """
    if depth == 0:
        return DecisionNode(get_count_of_max_label(dataset, w))
    if not dataset:
        return DecisionNode(get_count_of_max_label(old_dataset))
    elif len(attributes) <= 0:
        return DecisionNode(get_count_of_max_label(dataset))
    elif same_class(dataset):
        return DecisionNode(dataset[0][-1])
    else:
        best_attribute = get_best_split(dataset, attributes, w)
        if best_attribute is None:
            return DecisionNode(get_count_of_max_label(dataset))

        root = DecisionNode(best_attribute)

        true_dataset = []
        false_dataset = []
        for i in range(len(dataset)):
            if eval(dataset[i][int(best_attribute)]):
                true_dataset.append(dataset[i])
            else:
                false_dataset.append(dataset[i])

        new_attributes = list(filter(lambda x: x != best_attribute, attributes))

        root.true = decision_tree(true_dataset, new_attributes, dataset, depth - 1, w)
        root.false = decision_tree(false_dataset, new_attributes, dataset, depth - 1, w)

        return root
