import re


def beginning_with_ge(line):
    """
    Common Dutch words begin with 'ge' which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if l.startswith("ge"):
            return "True"

    return "False"


def occurrence_of_aa(line):
    """
    Common Dutch words contain 'aa' as a substring which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if 'aa' in l:
            return "True"

    return "False"


def containing_sch(line):
    """
    Common Dutch words contain 'sch' as a substring which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if "sch" in l:
            return "True"
    return "False"


def containing_lijk(line):
    """
    Common Dutch words contain 'lijk' as a substring which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if "lijk" in l:
            return "True"
    return "False"


def containing_cht(line):
    """
    Common Dutch words contain 'cht' as a substring which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if "cht" in l:
            return "True"
    return "False"


def containing_en(line):
    """
    Dutch word 'en' is very common.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if "en" == l:
            return "True"
    return "False"


def english_articles(line):
    """
    Common English language articles.
    :param line: input dataset line
    :return: True or False.
    """
    articles = ["the", "an", "a"]
    for l in line.split(" "):
        if l in articles:
            return "True"

    return "False"


def dutch_articles(line):
    """
    Common Dutch language articles.
    :param line: input dataset line
    :return: True or False.
    """
    articles = ["een", "de", "het"]
    for l in line.split(" "):
        if l in articles:
            return "True"

    return "False"


def ending_with_ig(line):
    """
    Common Dutch words end with 'ig' which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if l.endswith("ig"):
            return "True"
    return "False"


def average_word_length(line):
    """
    Average word length of Dutch language is greater than that of English language.
    :param line: input dataset line
    :return: True or False.
    """
    sum = 0
    count = 0
    for l in line.split(" "):
        sum += len(l)
        count += 1

    return "True" if sum / count > 8 else "False"


def dutch_common_words(line):
    """
    Some commonly used Dutch words.
    :param line: input dataset line
    :return: True or False.
    """
    words = ["aan", "dat", "en", "te", "voor"]
    for l in line.split(" "):
        if l in words:
            return "True"
    return "False"


def english_common_words(line):
    """
    Some commonly used English words.
    :param line: input dataset line
    :return: True or False.
    """
    words = ["the", "of", "to", "for", "at", "and", "so", "as"]
    for l in line.split(" "):
        if l in words:
            return "True"
    return "False"


def containing_kt(line):
    """
    Common Dutch words contain 'kt' as a substring which is not the case with English language.
    :param line: input dataset line
    :return: True or False.
    """
    for l in line.split(" "):
        if "kt" in l:
            return "True"

    return "False"


def extract(line):
    """
    Helper Function to extract features from each 15 word sentence fragment.
    :param line:
    :return:
    """
    feature = ["False"] * 13

    # attribute 1
    feature[0] = beginning_with_ge(line)

    # attribute 2
    feature[1] = occurrence_of_aa(line)

    # attribute 3
    feature[2] = containing_sch(line)

    # attribute 4
    feature[3] = containing_cht(line)

    # attribute 5
    feature[4] = containing_lijk(line)

    # attribute 6
    feature[5] = containing_en(line)

    # attribute 7
    feature[6] = english_articles(line)

    # attribute 8
    feature[7] = dutch_articles(line)

    # attribute 9
    feature[8] = ending_with_ig(line)

    # attribute 10
    feature[9] = average_word_length(line)

    # attribute 11
    feature[10] = dutch_common_words(line)

    # attribute 12
    feature[11] = english_common_words(line)

    # attribute 13
    feature[12] = containing_kt(line)

    return feature


def get_attributes(input_data):
    """
    Function to get attributes from the dataset which can be used by the models.
    :param input_data: Raw data from the files
    :return: extracted features from each sentence.
    """
    attributes = []
    for line in input_data:
        line = line.split("|")
        if len(line) > 1:
            data = re.sub(r'([^a-zA-Z ]+?)', '', line[1])
        else:
            data = re.sub(r'([^a-zA-Z ]+?)', '', line[0])

        result = extract(data.lower())

        if len(line) > 1:
            result.append(line[0])

        attributes.append(result)

    return attributes

