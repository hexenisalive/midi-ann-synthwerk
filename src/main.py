# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
from random import uniform
from pprint import pprint
from math import ceil

from file import load_file, stream_to_file
from decode_data import data_to_stream
from encode_data import prepare_data
from w2v import build_word2vec_model


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
    return batch_tensor


def test_stream_to_file():
    # randomly generated simulated data for testing stream generation
    test_output_data = []
    for no_of_notes in range(128):
        test_output_data.append([uniform(-1, 3), uniform(-0.5, 0.5), uniform(0, 0.5)])

    stream = data_to_stream(test_output_data)
    stream_to_file(stream, "../test_output.mid")


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    prepare_data('../MIDIs/tschai/')
    build_word2vec_model()

    test_stream_to_file()
    input_data = prepare_tensors()

    # unfinished model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, input_shape=(128, 3), return_sequences=False))
    
    model.summary()

    # model.fit(input_data, input_data)



