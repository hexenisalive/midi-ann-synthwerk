# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf

from encode_data import prepare_data
from w2v import build_word2vec_model
from prepare_inputs import prepare_tensors


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    prepare_data('../MIDIs/tschai/')
    build_word2vec_model()

    input_data = prepare_tensors()

    # unfinished model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, input_shape=(128, 3), return_sequences=False))
    
    model.summary()

    # model.fit(input_data, input_data)



