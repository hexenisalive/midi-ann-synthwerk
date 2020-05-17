import numpy as np
import music21 as mu
import copy

from file import load_file, stream_to_file
from utilities import gen_timestamp


def data_to_stream(data):
    """
    Generate music21.stream basing on global dictionary and vocabulary generated through word2vec.

    :param data: List of notes represented by lists of 3 numbers [X, Y, offset]
    :return music21.stream:
    """
    master_dict = load_file("dictionary")
    vocab_dict = load_file("w2v_vocab")
    coords = np.asarray(vocab_dict["coords"])
    words = vocab_dict["words"]

    stream = mu.stream.Stream()
    print("building stream...")
    # iterates through each note/chord represented by [X, Y, offset]
    curr_offset = 0.0
    for new_id, element in enumerate(data):
        # Makes a deep copy of an element stored in global dictionary.
        # An existing dictionary key is given by "name_closest_coord".
        # It must be done outside the global dictionary to speed up the process
        # by not iterating through the whole dictionary
        curr_offset += (np.round(element[2]*4.0))/4.0
        dict_key = name_closest_coord([element[0], element[1]], coords, words)
        insert = copy.deepcopy(master_dict[dict_key]["element"])
        insert.id = str(new_id)
        insert.offset = curr_offset
        try:
            stream.insert(insert)
        except mu.exceptions21.StreamException as msg:
            print("an error has occurred:", msg, "skipping...")
    return stream


def name_closest_coord(coord, coords, words):
    """
    Get the name of the element whose coordinate is the closest to one given.

    :param coord: User defined coordinates.
    :param vocab: Vocabulary dictionary
    :return string:
    """
    dist = np.sum((coords - coord)**2, axis=1)
    return words[int(np.argmin(dist))]


def prepare_output(data, iteration, tag=""):
    stream = data_to_stream(data)
    stream_to_file(stream, '../outputs/output_' + gen_timestamp() + '_' + tag + '_' + str(iteration) + '.mid')
