# MIDI files copyright holder
# Name: Bernd Krueger
# Source: http://www.piano-midi.de


import tensorflow as tf
import numpy as np
from os import listdir
from encode_data import prepare_data
from w2v import build_word2vec_model
from prep_learn_data import prepare_tensors
from decode_data import prepare_output
from utilities import prompt_question, gen_noise_vector, \
    gen_zero_vector, gen_partial_vector, gen_timestamp, \
    rectify_vector
from file import load_file


if __name__ == "__main__":
    # prepare_data for : chopin / beeth / tschai / c_major-small / c_major
    if prompt_question("Build new input data dictionaries?"):
        prepare_data('..\MIDIs\\chopin\\')
        build_word2vec_model()
        input_data, target_data = prepare_tensors()
    else:
        print('skipping...')
        try:
            load_file('dictionary')
            load_file('sequences')
            load_file('w2v_vocab')
        except FileNotFoundError:
            print("Input data dictionaries not found. "
                  "Building new dictionaries.")
            prepare_data('..\MIDIs\\chopin\\')
            build_word2vec_model()
            input_data, target_data = prepare_tensors()
        try:
            input_data = load_file('input_tensor')
            target_data = load_file('target_tensor')
        except FileNotFoundError:
            print("Tensors not found. "
                  "Building new tensors.")
            input_data, target_data = prepare_tensors()

    if prompt_question("Build new model?"):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(
            3, input_shape=(128, 3), return_sequences=True))
        model.add(tf.keras.layers.LSTM(
            3, input_shape=(128, 3), return_sequences=True))
        model.add(tf.keras.layers.LSTM(
            3, input_shape=(128, 3), return_sequences=True))
        model.add(tf.keras.layers.Dense(
            3, tf.keras.layers.Activation('relu')))
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(input_data, target_data, epochs=10, verbose=1)

        if prompt_question("Save model?"):
            model.save('..\models\model_' + gen_timestamp())
    else:
        print('skipping...')

    if prompt_question("Build sequences on known models?"):
        noise_vec = gen_noise_vector()
        zero_vec = gen_zero_vector()
        partial_vec = gen_partial_vector(length=64, offset=32)

        prepare_output(np.reshape(noise_vec, (128, 3)), 0, "noise")
        prepare_output(np.reshape(zero_vec, (128, 3)), 0, "zero")
        prepare_output(np.reshape(partial_vec, (128, 3)), 0, "partial")

        vocab = load_file("w2v_vocab")
        coords = np.asarray(vocab["coords"])

        noise_vec = rectify_vector(noise_vec, coords)
        zero_vec = rectify_vector(zero_vec, coords)
        partial_vec = rectify_vector(partial_vec, coords)


        for directory in listdir('..\models'):
            print("processing: %s..." % directory)
            model = tf.keras.models.load_model('../models/' + directory)

            noise_vec_pred = model.predict(noise_vec)
            zero_vec_pred = model.predict(zero_vec)
            partial_vec_pred = model.predict(partial_vec)

            prepare_output(np.reshape(noise_vec_pred, (128, 3)), 0, directory + "_noise")
            prepare_output(np.reshape(zero_vec_pred, (128, 3)), 0, directory + "_zero")
            prepare_output(np.reshape(partial_vec_pred, (128, 3)), 0, directory + "_partial")

    else:
        print('skipping...')


