# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
from random import uniform
from pprint import pprint


from file import stream_to_file
from decode_data import data_to_stream
from encode_data import prepare_data
from w2v import build_word2vec_model
from prepare_inputs import prepare_tensors


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



