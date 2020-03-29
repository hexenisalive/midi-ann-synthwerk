import numpy as np
import music21 as mu
import copy
from file import load_file


def data_to_stream(data):
    """
    Generate music21.stream basing on global dictionary and vocabulary generated through word2vec.

    :param data: List of notes represented by lists of 3 numbers [X, Y, offset]
    :return music21.stream:
    """
    master_dict = load_file("dictionary")

    stream = mu.stream.Stream()
    print("building stream...")
    # iterates through each note/chord represented by [X, Y, offset]
    curr_offset = 0.0
    for new_id, element in enumerate(data):
        # Makes a deep copy of an element stored in global dictionary.
        # An existing dictionary key is given by "name_closest_coord".
        # It must be done outside the global dictionary to speed up the process
        # by not iterating through the whole dictionary
        curr_offset += normalize_offset(element[2])
        dict_key = name_closest_coord([element[0], element[1]])
        insert = copy.deepcopy(master_dict[dict_key]["element"])
        insert.id = str(new_id)
        insert.offset = curr_offset
        try:
            stream.insert(insert)
        except mu.exceptions21.StreamException as msg:
            print("an error has occurred:", msg, "skipping...")
    return stream


def name_closest_coord(coord):
    """
    Get the name of the element whose coordinate is the closest to one given.

    :param coord: User defined coordinates.
    :return string:
    """
    vocab = load_file("w2v_vocab")
    coords = vocab["coords"]
    words = vocab["words"]
    coords = np.asarray(coords)
    dist = np.sum((coords - coord)**2, axis=1)
    return words[int(np.argmin(dist))]


def normalize_offset(offset):
    return (np.round(offset*4.0))/4.0
