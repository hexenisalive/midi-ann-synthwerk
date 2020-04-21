import music21 as mu
from os import listdir

from file import save_file, file_to_stream

master_dict = {}
sequence_data = []
offset_data = []


def element_to_key(element):
    """
    Creates dictionary key names basing on type of element supplied.

    :param element: music21.note or music21.chord
    :return string:
    """
    key = ""
    if type(element) == mu.chord.Chord:
        key = "Chord"
        current_chord = element.sortAscending()
        for pitch in current_chord.pitches:
            key += str(pitch)
    else:
        key += "Note" + str(element.pitch)
    return key


def stream_to_data(stream):
    """
    Processes music21.stream subclass into readable data.

    Updates master dictionary with new variations of chords and notes.
    Fills data sequence and offset data sets.

    :param stream: Trimmed and sorted music21.stream consisting only notes and chords.
    :return None:
    """
    temp_sequence = []
    temp_offset = []
    for element in stream:
        dict_name = element_to_key(element)
        temp_sequence.append(dict_name)
        temp_offset.append(float(element.offset))
        if dict_name not in master_dict.keys():
            element.offset = 0
            element.volume = mu.volume.Volume(velocity=100)
            element.duration = mu.duration.Duration(1)
            element.id = "none"
            master_dict[str(dict_name)] = {"element": element}
    sequence_data.append(temp_sequence)
    offset_data.append(temp_offset)


def prepare_data(path):
    """
    Prepares MIDI data for word2vec model and further training.

    Iterates through each file in given path and extracts new notes and keys,
    adds new elements to global dictionary,
    sequences and corresponding offsets are stored in separate global lists.
    Builds a word2vec model that creates vocabulary for vectorization purposes.

    :param path: Path to directory containing MIDIs.
    :param plot: Bool parameter, set True if you want Word2Vec 2d projection to be plotted.
    :param annotate: Bool parameter, set True if you want annotations added to nodes in 2d projection.
    :return None:
    """
    for file in listdir(path):
        print("processing: %s..." %file)
        s = file_to_stream(path+file)
        stream_to_data(s)

    # saving all data
    print("saving dictionary...")
    save_file("dictionary", master_dict)

    print("saving sequences...")
    sequence_dict = {}
    sequence_dict.update({"sequences": sequence_data, "offsets": offset_data})
    save_file("sequences", sequence_dict)

