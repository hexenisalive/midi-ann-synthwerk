# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from random import uniform
from pprint import pprint
from math import ceil

from file import load_file, save_file, stream_to_file
from decode_data import data_to_stream
from encode_data import prepare_data


def build_word2vec_model(plot: bool = False, annotate: bool = False):
    print("building model...")

    master_dict = load_file("dictionary")
    sequence_dict = load_file("sequences")
    sequence_data = sequence_dict["sequences"]
    # based on sequence data generate word embeddings
    # where sentences are sequences of dictionary keys ('ChordG5G6', 'NoteD7', etc.)
    w2v_model = Word2Vec(sequence_data, min_count=1)

    # projecting vectors onto 2d space
    X = w2v_model[w2v_model.wv.vocab]
    coords = PCA(n_components=2).fit_transform(X)

    # adding coordinates to corresponding keys in global dictionary
    words = list(w2v_model.wv.vocab)
    for coord, word in zip(coords, words):
        master_dict[word]["coords"] = list(coord)

    print("saving updated dictionary...")
    save_file("dictionary", master_dict)

    vocab = {"coords": list(coords), "words": words}
    save_file("w2v_vocab", vocab)

    # optional code snippet that plots the word embedding in 2d space
    if bool(plot):
        plt.scatter(coords[:, 0], coords[:, 1])
        if bool(annotate):
            for i, word in enumerate(words):
                plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
        plt.show()


def prepare_tensors():
    batch_tensor_list = []

    master_dict = load_file('dictionary')
    sequence_dict = load_file('sequences')
    sequence_data = sequence_dict["sequences"]
    offset_data = sequence_dict["offsets"]

    for sequence, offset in zip(sequence_data, offset_data):
        tensor_list = []
        curr_offset = 0.0
        for i, (element_s, element_o) in enumerate(zip(sequence, offset)):
            tensor_list.append(tf.Variable([
                master_dict[element_s]["coords"][0],
                master_dict[element_s]["coords"][1],
                element_o - curr_offset],
                tf.float32))
            curr_offset = element_o
        for trim in range(ceil(len(tensor_list)/128)):
            if (trim + 1) * 128 > len(tensor_list):
                batch_tensor_list.append(tf.stack(tensor_list[len(tensor_list)-128: len(tensor_list)]))
            else:
                batch_tensor_list.append(tf.stack(tensor_list[0 + (trim * 128): 128 + (trim * 128)]))
    batch_tensor = tf.stack(batch_tensor_list)
    print(batch_tensor.shape)
    return batch_tensor


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    prepare_data('../MIDIs/tschai/')
    build_word2vec_model()

    input_data = prepare_tensors()

    """ 
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, input_shape=(128, 3), return_sequences=False))
    
    model.summary()
    """

    """
    # randomly generated output data for testing stream generation
    test_output_data = []
    for no_of_notes in range(128):
        test_output_data.append([uniform(-1, 3), uniform(-0.5, 0.5), no_of_notes*0.5])
    
    stream = data_to_stream(test_output_data)
    stream_to_file(stream, "../test_output.mid")
    
    """
