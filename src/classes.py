
class Vocab:
    def __init__(self, coords_, words_):
        self.coords = coords_
        self.words = words_


class SequenceNode:
    def __init__(self, name_, offset_):
        self.name = name_
        self.offset = offset_