# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
import numpy as np

from encode_data import prepare_data
from w2v import build_word2vec_model
from prepare_inputs import prepare_tensors
from decode_data import prepare_output
from utilities import prompt_question, gen_noise_vector, gen_timestamp

if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai
    if prompt_question("Build new input data dictionaries? (Make sure files 'dictionary.pkl', "
                       "'sequences.pkl', 'w2v_vocab.pkl' exist in directory './pkl_files'.)n"):
        prepare_data('../MIDIs/tschai/')
        build_word2vec_model()

    input_data = prepare_tensors()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(3, input_shape=(128, 3), return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(input_data, input_data, epochs=100, verbose=2)

    test_data = gen_noise_vector()
    test_data = np.reshape(test_data, (1, 128, 3))

    prediction = model.predict(test_data)
    prepare_output(prediction)
    model.save('../models/model_' + gen_timestamp())

    print("all done")

