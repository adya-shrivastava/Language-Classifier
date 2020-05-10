class DecisionNode:
    __slots__ = "attribute", "true", "false"

    def __init__(self, attribute, true=None, false=None):
        self.attribute = attribute
        self.true = true
        self.false = false


class AdaBoostNode:

    __slots__ = "h", "z"

    def __init__(self, h, z):
        self.h = h
        self.z = z
